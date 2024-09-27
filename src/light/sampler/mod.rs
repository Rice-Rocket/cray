use std::sync::Arc;

use bvh::BvhLightSampler;
use uniform::UniformLightSampler;

use crate::{error, reader::target::FileLoc, Float};

use super::{Light, LightSampleContext};

pub mod uniform;
pub mod bvh;

pub trait AbstractLightSampler {
    fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight>;

    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float;

    fn sample_light(&self, u: Float) -> Option<SampledLight>;

    fn pmf_light(&self, light: &Light) -> Float;
}

pub struct SampledLight {
    pub light: Arc<Light>,
    /// Discrete probability for this light to be sampled
    pub p: Float,
}

pub enum LightSampler {
    Uniform(UniformLightSampler),
    Bvh(BvhLightSampler),
}

impl AbstractLightSampler for LightSampler {
    fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        match self {
            LightSampler::Uniform(s) => s.sample(ctx, u),
            LightSampler::Bvh(s) => s.sample(ctx, u),
        }
    }

    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float {
        match self {
            LightSampler::Uniform(s) => s.pmf(ctx, light),
            LightSampler::Bvh(s) => s.pmf(ctx, light),
        }
    }

    fn sample_light(&self, u: Float) -> Option<SampledLight> {
        match self {
            LightSampler::Uniform(s) => s.sample_light(u),
            LightSampler::Bvh(s) => s.sample_light(u),
        }
    }

    fn pmf_light(&self, light: &Light) -> Float {
        match self {
            LightSampler::Uniform(s) => s.pmf_light(light),
            LightSampler::Bvh(s) => s.pmf_light(light),
        }
    }
}

impl LightSampler {
    pub fn create(name: &str, lights: Arc<[Arc<Light>]>) -> LightSampler {
        match name {
            "uniform" => LightSampler::Uniform(UniformLightSampler {
                lights,
            }),
            "bvh" => LightSampler::Bvh(BvhLightSampler::new(lights)),
            _ => { error!(@basic "unknown light sampler: '{}'", name); },
        }
    }
}
