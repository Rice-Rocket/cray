use bumpalo::Bump;
use rand::rngs::SmallRng;

use crate::{bsdf::BSDF, bxdf::{BxDFReflTransFlags, TransportMode}, camera::Camera, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, interaction::SurfaceInteraction, light::{sampler::{AbstractLightSampler, LightSampler}, AbstractLight, LightSampleContext}, options::Options, sampler::{AbstractSampler, Sampler}, sampling::power_heuristic, Dot, Float, RayDifferential};

use super::{AbstractRayIntegrator, IntegratorBase};

pub struct PathIntegrator {
    max_depth: i32,
    light_sampler: LightSampler,
    regularize: bool,
}

impl PathIntegrator {
    pub fn new(max_depth: i32, light_sampler: LightSampler, regularize: bool) -> PathIntegrator {
        PathIntegrator {
            max_depth,
            light_sampler,
            regularize,
        }
    }

    fn sample_ld(
        &self,
        base: &IntegratorBase,
        intr: &SurfaceInteraction,
        bsdf: &BSDF,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
    ) -> SampledSpectrum {
        let mut ctx = LightSampleContext::from(intr);

        let flags = bsdf.flags();
        if flags.is_reflective() && !flags.is_transmissive() {
            ctx.pi = intr.interaction.offset_ray_origin_d(intr.interaction.wo).into();
        } else if flags.is_transmissive() && !flags.is_reflective() {
            ctx.pi = intr.interaction.offset_ray_origin_d(-intr.interaction.wo).into();
        }

        let u = sampler.get_1d();
        let Some(sampled_light) = self.light_sampler.sample(&ctx, u) else {
            return SampledSpectrum::from_const(0.0);
        };
        debug_assert!(sampled_light.p > 0.0);

        let u_light = sampler.get_2d();
        let light = sampled_light.light;
        let Some(ls) = light.sample_li(&ctx, u_light, lambda, true) else {
            return SampledSpectrum::from_const(0.0);
        };

        if ls.l.is_zero() || ls.pdf == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let wo = intr.interaction.wo;
        let wi = ls.wi;
        let f = bsdf.f(wo, wi, TransportMode::Radiance) * wi.dot(intr.shading.n).abs();
        if f.is_zero() || !base.unoccluded(&intr.interaction, &ls.p_light) {
            return SampledSpectrum::from_const(0.0);
        }

        let p_l = sampled_light.p * ls.pdf;
        if light.light_type().is_delta() {
            ls.l * f / p_l
        } else {
            let p_b = bsdf.pdf(wo, wi, TransportMode::Radiance, BxDFReflTransFlags::ALL);
            let w_l = power_heuristic(1, p_l, 1, p_b);
            w_l * ls.l * f / p_l
        }
    }
}

impl AbstractRayIntegrator for PathIntegrator {
    fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum {
        let mut l = SampledSpectrum::from_const(0.0);
        let mut beta = SampledSpectrum::from_const(1.0);
        let mut depth = 0;

        let mut p_b = 1.0;
        let mut eta_scale = 1.0;

        let mut specular_bounce = false;
        let mut any_non_specular_bounces = false;
        let mut prev_intr_ctx = LightSampleContext::default();

        loop {
            let Some(mut si) = base.intersect(&ray.ray, Float::INFINITY) else {
                for light in base.infinite_lights.iter() {
                    let le = light.le(&ray.ray, lambda);
                    if depth == 0 || specular_bounce {
                        l += beta * le;
                    } else {
                        let p_l = self.light_sampler.pmf(&prev_intr_ctx, light)
                            * light.pdf_li(&prev_intr_ctx, ray.ray.direction, true);
                        let w_b = power_heuristic(1, p_b, 1, p_l);
                        l += beta * w_b * le;
                    }
                }

                break;
            };

            let le = si.intr.le(-ray.ray.direction, lambda);
            if !le.is_zero() {
                if depth == 0 || specular_bounce {
                    l += beta * le;
                } else if let Some(light) = &si.intr.area_light {
                    let p_l = self.light_sampler.pmf(&prev_intr_ctx, light)
                        * light.pdf_li(&prev_intr_ctx, ray.ray.direction, true);
                    let w_l = power_heuristic(1, p_b, 1, p_l);
                    l += beta * w_l * le;
                }
            }

            let Some(mut bsdf) = si.intr.get_bsdf(ray, lambda, camera, sampler, options, rng) else {
                specular_bounce = true;
                si.intr.skip_intersection(ray, si.t_hit);
                continue;
            };

            if self.regularize && any_non_specular_bounces {
                bsdf.regularize();
            }

            if depth == self.max_depth {
                break;
            }

            depth += 1;

            if bsdf.flags().is_non_specular() {
                let ld = self.sample_ld(base, &si.intr, &bsdf, lambda, sampler);
                l += beta * ld;
            }

            let wo = -ray.ray.direction;
            let u = sampler.get_1d();
            let Some(bs) = bsdf.sample_f(
                wo,
                u,
                sampler.get_2d(),
                TransportMode::Radiance,
                BxDFReflTransFlags::all(),
            ) else {
                break;
            };

            beta *= bs.f * bs.wi.dot(si.intr.shading.n).abs() / bs.pdf;
            p_b = if bs.pdf_is_proportional {
                bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BxDFReflTransFlags::all())
            } else {
                bs.pdf
            };

            debug_assert!(beta.y(lambda).is_finite());
            specular_bounce = bs.is_specular();
            any_non_specular_bounces |= !specular_bounce;
            if bs.is_transmission() {
                eta_scale *= bs.eta * bs.eta;
            }

            prev_intr_ctx = LightSampleContext::from(&si.intr);

            *ray = si.intr.spawn_ray_with_differentials(ray, bs.wi, bs.flags, bs.eta);
            
            if eta_scale.is_finite() {
                let rr_beta = beta * eta_scale;
                if rr_beta.max_component_value() < 1.0 && depth > 1 {
                    let q = Float::max(0.0, 1.0 - rr_beta.max_component_value());
                    if sampler.get_1d() < q {
                        break;
                    }
                    beta /= 1.0 - q;
                    debug_assert!(beta.y(lambda).is_finite());
                }
            }
        }

        l
    } 
}
