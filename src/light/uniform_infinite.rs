use std::sync::Arc;

use crate::{color::{sampled::SampledSpectrum, spectrum::{AbstractSpectrum, DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, interaction::Interaction, sampling::{sample_uniform_sphere, uniform_hemisphere_pdf, uniform_sphere_pdf}, transform::Transform, Bounds3f, Float, Normal3f, Point2f, Point3f, Ray, Vec3f, PI};

use super::{AbstractLight, LightBase, LightBounds, LightLiSample, LightSampleContext, LightType};

#[derive(Debug, Clone)]
pub struct UniformInfiniteLight {
    base: LightBase,
    l_emit: Arc<DenselySampledSpectrum>,
    scale: Float,
    scene_center: Point3f,
    scene_radius: Float,
}

impl UniformInfiniteLight {
    pub fn new(
        render_from_light: Transform,
        le: Arc<Spectrum>,
        scale: Float,
    ) -> UniformInfiniteLight {
        let base = LightBase {
            ty: LightType::Infinite,
            render_from_light,
            medium: None,
        };

        UniformInfiniteLight {
            base,
            l_emit: Arc::new(DenselySampledSpectrum::new(le.as_ref())),
            scale,
            scene_center: Point3f::ZERO,
            scene_radius: 0.0,
        }
    }
}

impl AbstractLight for UniformInfiniteLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        3.0 * PI * PI * self.scene_radius * self.scene_radius * self.scale * self.l_emit.sample(lambda)
    }

    fn light_type(&self) -> LightType {
        self.base.ty
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        if allow_incomplete_pdf {
            None
        } else {
            let wi = sample_uniform_sphere(u);
            let pdf = uniform_hemisphere_pdf();
            Some(LightLiSample::new(
                self.scale * self.l_emit.sample(lambda),
                wi,
                pdf,
                Interaction::new(
                    (ctx.p() + wi * (2.0 * self.scene_radius)).into(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    Default::default(),
                )
            ))
        }
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        if allow_incomplete_pdf {
            0.0
        } else {
            uniform_sphere_pdf()
        }
    }

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        self.base.l(p, n, uv, w, lambda)
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.scale * self.l_emit.sample(lambda)
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        let bounding_sphere = scene_bounds.bounding_sphere();
        self.scene_center = bounding_sphere.0;
        self.scene_radius = bounding_sphere.1;
    }

    fn bounds(&self) -> Option<LightBounds> {
        None
    }
}
