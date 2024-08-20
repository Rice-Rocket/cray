use bumpalo::Bump;
use rand::rngs::SmallRng;

use crate::{bxdf::{BxDFReflTransFlags, TransportMode}, camera::Camera, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, light::{sampler::{uniform::UniformLightSampler, AbstractLightSampler}, AbstractLight, LightSampleContext}, options::Options, sampler::{AbstractSampler, Sampler}, sampling::{sample_uniform_hemisphere, sample_uniform_sphere, uniform_hemisphere_pdf, uniform_sphere_pdf}, Dot, Float, RayDifferential};

use super::{AbstractRayIntegrator, IntegratorBase};

pub struct SimplePathIntegrator {
    pub max_depth: i32,
    pub sample_lights: bool,
    pub sample_bsdf: bool,
    pub light_sampler: UniformLightSampler,
}

impl AbstractRayIntegrator for SimplePathIntegrator {
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
        let mut specular_bounce = true;

        let mut beta = SampledSpectrum::from_const(1.0);
        let mut depth = 0;

        while !beta.is_zero() {
            let Some(si) = base.intersect(&ray.ray, Float::INFINITY) else {
                if !self.sample_lights || specular_bounce {
                    for light in base.infinite_lights.iter() {
                        l += beta * light.le(&ray.ray, lambda);
                    }
                }
                break;
            };

            let mut isect = si.intr;
            if !self.sample_lights || specular_bounce {
                l += beta * isect.le(-ray.ray.direction, lambda);
            }

            if depth == self.max_depth {
                break;
            }

            depth += 1;

            let Some(bsdf) = isect.get_bsdf(ray, lambda, camera, sampler, options, rng) else {
                specular_bounce = true;
                isect.skip_intersection(ray, si.t_hit);
                continue;
            };

            let wo = -ray.ray.direction;

            if self.sample_lights {
                let sampled_light = self.light_sampler.sample_light(sampler.get_1d());
                if let Some(sampled_light) = sampled_light {
                    let u_light = sampler.get_2d();
                    let light_sample_ctx = LightSampleContext::from(&isect);
                    let ls = sampled_light.light.sample_li(&light_sample_ctx, u_light, lambda, false);

                    if let Some(ls) = ls {
                        if !ls.l.is_zero() && ls.pdf > 0.0 {
                            let wi = ls.wi;
                            let f = bsdf.f(wo, wi, TransportMode::Radiance) * wi.dot(isect.shading.n).abs();
                            if !f.is_zero() && base.unoccluded(&isect.interaction, &ls.p_light) {
                                l += beta * f * ls.l / (sampled_light.p * ls.pdf);
                            }
                        }
                    }
                }
            }

            if self.sample_bsdf {
                let u = sampler.get_1d();
                let Some(bs) = bsdf.sample_f(
                    wo,
                    u,
                    sampler.get_2d(),
                    TransportMode::Radiance,
                    BxDFReflTransFlags::all(),
                ) else { break };

                beta *= bs.f * bs.wi.dot(isect.shading.n).abs() / bs.pdf;
                specular_bounce = bs.is_specular();
                *ray = isect.interaction.spawn_ray(bs.wi);
            } else {
                let flags = bsdf.flags();
                let (pdf, wi) = if flags.is_reflective() && flags.is_transmissive() {
                    let wi = sample_uniform_sphere(sampler.get_2d());
                    let pdf = uniform_sphere_pdf();
                    (pdf, wi)
                } else {
                    let wi = sample_uniform_hemisphere(sampler.get_2d());
                    let pdf = uniform_hemisphere_pdf();
                    let wi = if (flags.is_reflective() 
                        && wo.dot(isect.interaction.n) * wi.dot(isect.interaction.n) < 0.0)
                        || (flags.is_transmissive()
                        && wo.dot(isect.interaction.n) * wi.dot(isect.interaction.n) > 0.0)
                    {
                        -wi
                    } else {
                        wi
                    };

                    (pdf, wi)
                };
                beta *= bsdf.f(wo, wi, TransportMode::Radiance)
                    * wi.dot(isect.shading.n).abs() / pdf;
                specular_bounce = false;
                *ray = isect.interaction.spawn_ray(wi);
            }

            debug_assert!(beta.y(lambda) >= 0.0);
            debug_assert!(beta.y(lambda).is_finite());
        }

        debug_assert!(!l.values[0].is_nan());
        debug_assert!(!l.values[1].is_nan());
        debug_assert!(!l.values[2].is_nan());
        debug_assert!(!l.values[3].is_nan());

        l
    }
}
