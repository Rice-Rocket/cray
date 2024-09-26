use bumpalo::Bump;
use rand::{rngs::SmallRng, Rng};

use crate::{bxdf::{BxDFReflTransFlags, TransportMode}, camera::Camera, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, light::AbstractLight, media::{sample_t_maj, AbstractMedium, Medium, MediumProperties}, options::Options, phase::AbstractPhaseFunction, sampler::{AbstractSampler, Sampler}, sampling::sample_discrete, Float, Point2f, Point3f, Ray, RayDifferential};

use super::{AbstractRayIntegrator, IntegratorBase};

pub struct SimpleVolumetricPathIntegrator {
    pub(super) max_depth: i32,
}

impl AbstractRayIntegrator for SimpleVolumetricPathIntegrator {
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
        let mut beta = 1.0;
        let mut depth = 0;

        lambda.terminate_secondary();

        loop {
            let si = base.intersect(&ray.ray, Float::INFINITY);
            let mut scattered = false;
            let mut terminated = false;

            if let Some(ref m) = ray.ray.medium {
                let t_max = si.as_ref().map(|s| s.t_hit).unwrap_or(Float::INFINITY);
                let u = sampler.get_1d();
                let mut u_mode = sampler.get_1d();
                
                sample_t_maj(
                    &mut ray.ray,
                    t_max,
                    u,
                    rng,
                    lambda,
                    |ray: &mut Ray, p: Point3f, mp: &MediumProperties, sigma_maj: &SampledSpectrum, t_maj: &SampledSpectrum, rng: &mut SmallRng| {
                        let p_absorb = mp.sigma_a[0] / sigma_maj[0];
                        let p_scatter = mp.sigma_s[0] / sigma_maj[0];
                        let p_null = Float::max(0.0, 1.0 - p_absorb - p_scatter);

                        let mode = sample_discrete(&[p_absorb, p_scatter, p_null], u_mode, None, None);
                        if mode == Some(0) {
                            l += beta * mp.le;
                            terminated = true;
                            false
                        } else if mode == Some(1) {
                            if depth >= self.max_depth {
                                depth += 1;
                                terminated = true;
                                return false;
                            }

                            depth += 1;

                            let u = Point2f::new(rng.gen(), rng.gen());

                            let Some(ps) = mp.phase.sample_p(-ray.direction, u) else {
                                terminated = true;
                                return false;
                            };

                            beta *= ps.p / ps.pdf;
                            ray.origin = p;
                            ray.direction = ps.wi;
                            scattered = true;
                            false
                        } else {
                            u_mode = rng.gen();
                            true
                        }
                    },
                );
            }

            if terminated {
                return l;
            }

            if scattered {
                continue;
            }

            let Some(mut si) = si else {
                for light in base.infinite_lights.iter() {
                    l += beta * light.le(&ray.ray, lambda);
                }

                return l;
            };

            l += beta * si.intr.le(-ray.ray.direction, lambda);

            if let Some(bsdf) = si.intr.get_bsdf(ray, lambda, camera, sampler, options, rng) {
                let uc = sampler.get_1d();
                let u = sampler.get_2d();
                if bsdf.sample_f(-ray.ray.direction, uc, u, TransportMode::Radiance, BxDFReflTransFlags::ALL).is_some() {
                    panic!("SimpleVolumetricPathIntegrator doesn't support surface scattering");
                } else {
                    break;
                }
            } else {
                si.intr.skip_intersection(ray, si.t_hit);
            }
        }

        l
    }
}
