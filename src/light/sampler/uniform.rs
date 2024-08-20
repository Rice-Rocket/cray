use std::sync::Arc;

use crate::{light::{Light, LightSampleContext}, Float};

use super::{AbstractLightSampler, SampledLight};

pub struct UniformLightSampler {
    pub lights: Arc<[Arc<Light>]>,
}

impl AbstractLightSampler for UniformLightSampler {
    fn sample(&self, _ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        self.sample_light(u)
    }

    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float {
        self.pmf_light(light)
    }

    fn sample_light(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }

        let light_index = usize::min(
            (u * self.lights.len() as Float) as usize,
            self.lights.len() - 1,
        );

        Some(SampledLight {
            light: self.lights[light_index].clone(),
            p: 1.0 / self.lights.len() as Float,
        })
    }

    fn pmf_light(&self, light: &Light) -> Float {
        if self.lights.is_empty() {
            0.0
        } else {
            1.0 / self.lights.len() as Float
        }
    }
}
