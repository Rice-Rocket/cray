use std::sync::Arc;

use bumpalo::Bump;
use rand::{rngs::SmallRng, Rng};

use crate::{bsdf::BSDF, bssrdf::{AbstractBSSRDF, SubsurfaceInteraction}, bxdf::{BxDFReflTransFlags, TransportMode}, camera::{film::VisibleSurface, Camera}, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, float_to_bits, hashing::mix_bits, interaction::{GeneralInteraction, Interaction, MediumInteraction, SurfaceInteraction}, light::{sampler::{AbstractLightSampler, LightSampler}, AbstractLight, LightSampleContext}, media::{sample_t_maj, MediumProperties}, options::Options, phase::AbstractPhaseFunction, reader::error::ParseResult, sampler::{AbstractSampler, Sampler}, sampling::{sample_discrete, WeightedReservoirSampler}, Dot, Float, Normal3f, Point2f, Point3f, Point3fi, Ray, RayDifferential, Vec3f};

use super::{AbstractRayIntegrator, IntegratorBase};

pub struct VolumetricPathIntegrator {
    max_depth: i32,
    light_sampler: LightSampler,
    regularize: bool,
}

impl VolumetricPathIntegrator {
    const SHADOW_EPSILON: Float = 0.0001;

    pub fn new(max_depth: i32, light_sampler: LightSampler, regularize: bool) -> ParseResult<VolumetricPathIntegrator> {
        Ok(VolumetricPathIntegrator {
            max_depth,
            light_sampler,
            regularize,
        })
    }

    pub fn sample_ld(
        &self,
        base: &IntegratorBase,
        intr: &GeneralInteraction,
        bsdf: Option<&BSDF>,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
        beta: &SampledSpectrum,
        r_p: &SampledSpectrum,
        rng: &mut SmallRng,
    ) -> SampledSpectrum {
        let ctx = if let Some(bsdf) = bsdf {
            let pi: Point3fi = if bsdf.flags().is_reflective() && !bsdf.flags().is_transmissive() {
                intr.intr().offset_ray_origin_d(intr.intr().wo).into()
            } else if bsdf.flags().is_transmissive() && !bsdf.flags().is_reflective() {
                intr.intr().offset_ray_origin_d(-intr.intr().wo).into()
            } else {
                intr.intr().pi
            };

            LightSampleContext::from(intr.as_surface())
        } else {
            LightSampleContext::from(intr.intr())
        };

        let u = sampler.get_1d();
        let u_light = sampler.get_2d();

        let Some(sampled_light) = self.light_sampler.sample(&ctx, u) else {
            return SampledSpectrum::from_const(0.0);
        };

        let light = &sampled_light.light;
        debug_assert!(sampled_light.p != 0.0);

        let Some(ls) = light.sample_li(&ctx, u_light, lambda, true) else {
            return SampledSpectrum::from_const(0.0);
        };

        if ls.l.is_zero() || ls.pdf == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let p_l = sampled_light.p * ls.pdf;
        let wo = intr.intr().wo;
        let wi = ls.wi;

        let (f_hat, scatter_pdf) = if let Some(bsdf) = bsdf {
            (
                bsdf.f(wo, wi, TransportMode::Radiance) * wi.dot(intr.as_surface().shading.n).abs(),
                bsdf.pdf(wo, wi, TransportMode::Radiance, BxDFReflTransFlags::ALL)
            )
        } else {
            debug_assert!(intr.intr().is_medium_interaction());
            let phase = intr.as_medium().phase;
            (
                SampledSpectrum::from_const(phase.p(wo, wi)),
                phase.pdf(wo, wi)
            )
        };

        if f_hat.is_zero() {
            return SampledSpectrum::from_const(0.0);
        }

        let mut light_ray = intr.intr().spawn_ray_to_interaction(&ls.p_light);
        let mut t_ray = SampledSpectrum::from_const(1.0);
        let mut r_l = SampledSpectrum::from_const(1.0);
        let mut r_u = SampledSpectrum::from_const(1.0);

        while light_ray.direction != Vec3f::ZERO {
            let si = base.intersect(&light_ray, 1.0 - Self::SHADOW_EPSILON);

            if si.as_ref().is_some_and(|s| s.intr.material.is_some()) {
                return SampledSpectrum::from_const(0.0);
            }

            if let Some(ref med) = light_ray.medium {
                let t_max = if let Some(ref s) = si { s.t_hit } else { 1.0 - Self::SHADOW_EPSILON };
                let u: Float = rng.gen();
                let t_maj = sample_t_maj(
                    &mut light_ray,
                    t_max,
                    u,
                    rng,
                    lambda,
                    |light_ray, p, mp, sigma_maj, t_maj, rng| {
                        let sigma_n = (sigma_maj - mp.sigma_a - mp.sigma_s).clamp_zero();
                        let pdf = t_maj[0] * sigma_maj[0];
                        t_ray *= t_maj * sigma_n / pdf;
                        r_l *= t_maj * sigma_maj / pdf;
                        r_u *= t_maj * sigma_n / pdf;

                        let tr = t_ray / (r_l + r_u).average();
                        if tr.max_component_value() < 0.05 {
                            let q = 0.75;
                            if rng.gen::<Float>() < q {
                                t_ray = SampledSpectrum::from_const(0.0);
                            } else {
                                t_ray /= 1.0 - q;
                            }
                        }

                        !t_ray.is_zero()
                    }
                );

                t_ray *= t_maj / t_maj[0];
                r_l *= t_maj / t_maj[0];
                r_u *= t_maj / t_maj[0];
            }

            if t_ray.is_zero() {
                return SampledSpectrum::from_const(0.0);
            }

            if let Some(ref s) = si {
                light_ray = s.intr.interaction.spawn_ray_to_interaction(&ls.p_light);
            } else {
                break;
            }
        }

        r_l *= r_p * p_l;
        r_u *= r_p * scatter_pdf;
        if light.light_type().is_delta() {
            beta * f_hat * t_ray * ls.l / r_l.average()
        } else {
            beta * f_hat * t_ray * ls.l / (r_l + r_u).average()
        }
    }
}

impl AbstractRayIntegrator for VolumetricPathIntegrator {
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
        let mut r_u = SampledSpectrum::from_const(1.0);
        let mut r_l = SampledSpectrum::from_const(1.0);

        let mut specular_bounce = false;
        let mut any_non_specular_bounce = false;
        let mut depth = 0;
        let mut eta_scale = 1.0;

        let mut prev_intr_context = LightSampleContext::default();

        loop {
            let si = base.intersect(&ray.ray, Float::INFINITY);

            if let Some(ref medium) = ray.ray.medium {
                let mut scattered = false;
                let mut terminated = false;
                let t_max = if let Some(ref s) = si { s.t_hit } else { Float::INFINITY };

                let t_maj = sample_t_maj(
                    &mut ray.ray,
                    t_max,
                    sampler.get_1d(),
                    rng,
                    lambda,
                    |ray, p, mp, sigma_maj, t_maj, rng| {
                        if beta.is_zero() {
                            terminated = true;
                            return false;
                        }

                        // volume_interactions += 1;

                        if depth < self.max_depth && !mp.le.is_zero() {
                            let pdf = sigma_maj[0] * t_maj[0];
                            let betap = beta * t_maj / pdf;

                            let r_e = r_u * sigma_maj * t_maj / pdf;

                            if !r_e.is_zero() {
                                l += betap * mp.sigma_a * mp.le / r_e.average();
                            }
                        }

                        let p_absorb = mp.sigma_a[0] / sigma_maj[0];
                        let p_scatter = mp.sigma_s[0] / sigma_maj[0];
                        let p_null = Float::max(0.0, 1.0 - p_absorb - p_scatter);

                        debug_assert!(1.0 - p_absorb - p_scatter >= -1e-6);
                        let um: Float = rng.gen();
                        let mode = sample_discrete(&[p_absorb, p_scatter, p_null], um, None, None);
                        if mode == Some(0) {
                            terminated = true;
                            false
                        } else if mode == Some(1) {
                            if depth >= self.max_depth {
                                depth += 1;
                                terminated = true;
                                return false;
                            }

                            depth += 1;

                            let pdf = t_maj[0] * mp.sigma_s[0];
                            beta *= t_maj * mp.sigma_s / pdf;
                            r_u *= t_maj * mp.sigma_s / pdf;

                            if !beta.is_zero() && !r_u.is_zero() {
                                let intr = MediumInteraction::new(p.into(), -ray.direction, ray.time, ray.medium.clone(), mp.phase);
                                l += self.sample_ld(base, &GeneralInteraction::Medium(intr.clone()), None, lambda, sampler, &beta, &r_u, rng);
                                let u = sampler.get_2d();
                                let ps = intr.phase.sample_p(-ray.direction, u);

                                if let Some(ps) = ps {
                                    if ps.pdf == 0.0 {
                                        terminated = true;
                                    } else {
                                        beta = beta * (ps.p / ps.pdf);
                                        r_l = r_u / ps.pdf;
                                        prev_intr_context = LightSampleContext::from(&intr.interaction);
                                        scattered = true;
                                        ray.origin = p;
                                        ray.direction = ps.wi;
                                        specular_bounce = false;
                                        any_non_specular_bounce = true;
                                    }
                                } else {
                                    terminated = true;
                                }
                            }

                            false
                        } else {
                            let sigma_n = (sigma_maj - mp.sigma_a - mp.sigma_s).clamp_zero();
                            let pdf = t_maj[0] * sigma_n[0];
                            beta *= t_maj * sigma_n / pdf;
                            if pdf == 0.0 {
                                beta = SampledSpectrum::from_const(0.0);
                            }

                            r_u *= t_maj * sigma_n / pdf;
                            r_l *= t_maj * sigma_maj / pdf;
                            !beta.is_zero() && !r_u.is_zero()
                        }
                    }
                );

                if terminated || beta.is_zero() || r_u.is_zero() {
                    return l;
                }

                if scattered {
                    continue;
                }

                beta *= t_maj / t_maj[0];
                r_u *= t_maj / t_maj[0];
                r_l *= t_maj / t_maj[0];
            }

            let Some(si) = si else {
                for light in base.infinite_lights.iter() {
                    let le = light.le(&ray.ray, lambda);
                    if !le.is_zero() {
                        if depth == 0 || specular_bounce {
                            l += beta * le / r_u.average();
                        } else {
                            let p_l = self.light_sampler.pmf(&prev_intr_context, light)
                                * light.pdf_li(&prev_intr_context, ray.ray.direction, true);

                            r_l = r_l * p_l;
                            l += beta * le / (r_u + r_l).average();
                        }
                    }
                }

                break;
            };

            let mut isect = si.intr;
            let le = isect.le(-ray.ray.direction, lambda);
            if !le.is_zero() {
                if depth == 0 || specular_bounce {
                    l += beta * le / r_u.average();
                } else {
                    let area_light = isect.area_light.as_ref().unwrap();
                    let p_l = self.light_sampler.pmf(&prev_intr_context, area_light)
                        * area_light.pdf_li(&prev_intr_context, ray.ray.direction, true);
                    
                    r_l = r_l * p_l;
                    l += beta * le / (r_u + r_l).average();
                }
            }

            let Some(mut bsdf) = isect.get_bsdf(ray, lambda, camera, sampler, options, rng) else {
                isect.skip_intersection(ray, si.t_hit);
                continue;
            };

            if depth >= self.max_depth {
                depth += 1;
                return l;
            }

            depth += 1;
            // surface_interactions += 1;

            if self.regularize && any_non_specular_bounce {
                // regularized_bsdf += 1;
                bsdf.regularize();
            }

            if bsdf.flags().is_non_specular() {
                l += self.sample_ld(base, &GeneralInteraction::Surface(isect.clone()), Some(&bsdf), lambda, sampler, &beta, &r_u, rng);
                debug_assert!(l.y(lambda).is_finite());
            }

            prev_intr_context = LightSampleContext::from(&isect);

            let wo = isect.interaction.wo;
            let u = sampler.get_1d();
            let Some(bs) = bsdf.sample_f(wo, u, sampler.get_2d(), TransportMode::Radiance, BxDFReflTransFlags::ALL) else {
                break;
            };

            beta *= bs.f * bs.wi.dot(isect.shading.n).abs() / bs.pdf;
            if bs.pdf_is_proportional {
                r_l = r_u / bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BxDFReflTransFlags::ALL);
            } else {
                r_l = r_u / bs.pdf;
            }

            debug_assert!(beta.y(lambda).is_finite());

            specular_bounce = bs.is_specular();
            any_non_specular_bounce |= !bs.is_specular();
            if bs.is_transmission() {
                eta_scale *= bs.eta * bs.eta;
            }
            
            *ray = isect.spawn_ray_with_differentials(ray, bs.wi, bs.flags, bs.eta);

            if let Some(bssrdf) = isect.get_bssrdf(ray, lambda, camera, rng) {
                if bs.is_transmission() {
                    let uc = sampler.get_1d();
                    let up = sampler.get_2d();
                    let Some(probe_seg) = bssrdf.sample_sp(uc, up) else { break };

                    let seed = mix_bits(float_to_bits(sampler.get_1d()) as u64);
                    let mut interaction_sampler: WeightedReservoirSampler<SubsurfaceInteraction> = 
                        WeightedReservoirSampler::new(seed, SubsurfaceInteraction::default());
                    let mut intr_base = Interaction::new(probe_seg.p0.into(), Normal3f::ZERO, Point2f::ZERO, Vec3f::ZERO, ray.ray.time);
                    loop {
                        let r = intr_base.spawn_ray_to(probe_seg.p1);
                        if r.direction == Vec3f::ZERO {
                            break;
                        }

                        let Some(si) = base.intersect(&r, 1.0) else { break };
                        intr_base = si.intr.interaction.clone();
                        if si.intr.material.is_some() && isect.material.is_some() {
                            if Arc::ptr_eq(si.intr.material.as_ref().unwrap(), si.intr.material.as_ref().unwrap()) {
                                interaction_sampler.add_sample(SubsurfaceInteraction::from(&si.intr), 1.0);
                            }
                        } else {
                            interaction_sampler.add_sample(SubsurfaceInteraction::from(&si.intr), 1.0);
                        }
                    }

                    if !interaction_sampler.has_sample() {
                        break;
                    }

                    let ssi = interaction_sampler.get_sample();
                    let mut bssrdf_sample = bssrdf.probe_intersection_to_sample(ssi);
                    if bssrdf_sample.sp.is_zero() || bssrdf_sample.pdf.is_zero() {
                        break;
                    }

                    let pdf = interaction_sampler.sample_probability() * bssrdf_sample.pdf[0];
                    beta *= bssrdf_sample.sp / pdf;
                    r_u *= bssrdf_sample.pdf / bssrdf_sample.pdf[0];
                    let mut pi = SurfaceInteraction::from(ssi);
                    pi.interaction.wo = bssrdf_sample.wo;
                    prev_intr_context = LightSampleContext::from(&pi);
                    any_non_specular_bounce = true;

                    if self.regularize {
                        // regularized_bsdf += 1;
                        bssrdf_sample.sw.regularize();
                    // } else {
                    //     total_bsdfs += 1;
                    }

                    l += self.sample_ld(base, &GeneralInteraction::Surface(pi.clone()), Some(&bssrdf_sample.sw), lambda, sampler, &beta, &r_u, rng);

                    let u = sampler.get_1d();
                    let Some(bs) = bssrdf_sample.sw.sample_f(
                        pi.interaction.wo,
                        u,
                        sampler.get_2d(),
                        TransportMode::Radiance,
                        BxDFReflTransFlags::ALL,
                    ) else {
                        break;
                    };

                    beta *= bs.f * bs.wi.dot(pi.shading.n).abs() / bs.pdf;
                    r_l = r_u / bs.pdf;
                    debug_assert!(beta.y(lambda).is_finite());
                    specular_bounce = bs.is_specular();
                    *ray = pi.interaction.spawn_ray(bs.wi);
                }
            };

            if beta.is_zero() {
                break;
            }

            let rr_beta = beta * eta_scale / r_u.average();
            let u_rr = sampler.get_1d();

            if rr_beta.max_component_value() < 1.0 && depth > 1 {
                let q = Float::max(0.0, 1.0 - rr_beta.max_component_value());
                if u_rr < q {
                    break;
                }

                beta /= 1.0 - q;
            }
        }

        l
    }
}
