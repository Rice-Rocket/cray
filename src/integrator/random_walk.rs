use bumpalo::Bump;
use rand::rngs::SmallRng;

use crate::{bxdf::TransportMode, camera::Camera, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, light::AbstractLight, options::Options, sampler::{AbstractSampler, Sampler}, sampling::sample_uniform_sphere, Dot, Float, RayDifferential, PI};

use super::{AbstractRayIntegrator, IntegratorBase};

pub struct RandomWalkIntegrator {
    pub max_depth: i32,
}

impl AbstractRayIntegrator for RandomWalkIntegrator {
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
        self.li_random_walk(
            base,
            camera,
            ray,
            lambda,
            sampler,
            0,
            scratch_buffer,
            options,
            rng,
        )
    }
}

impl RandomWalkIntegrator {
    fn li_random_walk(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        depth: i32,
        _scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum {
        let Some(mut si) = base.intersect(&ray.ray, Float::INFINITY) else {
            let mut le = SampledSpectrum::from_const(0.0);
            for light in base.infinite_lights.iter() {
                le += light.le(&ray.ray, lambda);
            }
            return le
        };

        let isect = &mut si.intr;

        let wo = -ray.ray.direction;
        let le = isect.le(wo, lambda);

        if depth == self.max_depth {
            return le;
        }

        let Some(bsdf) = isect.get_bsdf(ray, lambda, camera, sampler, options, rng) else { return le };

        let u = sampler.get_2d();
        let wp = sample_uniform_sphere(u);

        let f = bsdf.f(wo, wp, TransportMode::Radiance);
        if f.is_zero() {
            return le;
        }

        let fcos = f * wp.dot(isect.shading.n).abs();

        let ray = isect.interaction.spawn_ray(wp);

        le + fcos * self.li_random_walk(
            base,
            camera,
            &ray,
            lambda,
            sampler,
            depth + 1,
            _scratch_buffer,
            options,
            rng,
        ) / (1.0 / (4.0 * PI))
    }
}
