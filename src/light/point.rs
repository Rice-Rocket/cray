use std::{collections::HashMap, sync::Arc};

use crate::{color::{colorspace::RgbColorSpace, sampled::SampledSpectrum, spectrum::{spectrum_to_photometric, AbstractSpectrum, DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, error, interaction::Interaction, media::{Medium, MediumInterface}, reader::{error::ParseResult, paramdict::{ParameterDictionary, SpectrumType}, target::FileLoc}, transform::{ApplyTransform, Transform}, Bounds3f, Float, Normal3f, Point2f, Point3f, Ray, Vec3f, PI};

use super::{AbstractLight, LightBase, LightBounds, LightLiSample, LightSampleContext, LightType};

#[derive(Debug, Clone)]
pub struct PointLight {
    base: LightBase,
    i: Arc<DenselySampledSpectrum>,
    scale: Float,
}

impl PointLight {
    pub fn new(render_from_light: Transform, medium: Option<Arc<MediumInterface>>, i: Arc<Spectrum>, scale: Float) -> PointLight {
        let base = LightBase {
            ty: LightType::DeltaPosition,
            render_from_light,
            medium,
        };

        PointLight {
            base,
            i: Arc::new(DenselySampledSpectrum::new(&i)),
            scale,
        }
    }

    pub fn create(
        render_from_light: Transform,
        medium: Option<Arc<MediumInterface>>,
        parameters: &mut ParameterDictionary,
        color_space: Arc<RgbColorSpace>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<PointLight> {
        let i = parameters.get_one_spectrum(
            "I",
            Some(color_space.illuminant.clone()),
            SpectrumType::Illuminant,
            cached_spectra,
        )?.ok_or(error!(@create loc, MissingParameter, "point light requires 'i' parameter"))?;

        let mut sc = parameters.get_one_float("scale", 1.0)?;
        sc /= spectrum_to_photometric(&i);

        let phi_v = parameters.get_one_float("power", -1.0)?;
        if phi_v > 0.0 {
            let k_e = 4.0 * PI;
            sc *= phi_v / k_e;
        }

        let from = parameters.get_one_point3f("from", Point3f::ZERO)?;
        let tf = Transform::from_translation(from);
        let final_render_from_light = render_from_light.apply(tf);

        Ok(PointLight::new(final_render_from_light, medium, i, sc))
    }
}

impl AbstractLight for PointLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        4.0 * PI * self.scale * self.i.sample(lambda)
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
        let p = self.base.render_from_light.apply(Point3f::ZERO);
        let wi = (p - ctx.p()).normalize();
        let li = self.scale * self.i.sample(lambda) / p.distance_squared(ctx.p());

        Some(LightLiSample::new(
            li,
            wi.into(),
            1.0,
            Interaction {
                pi: p.into(),
                medium_interface: self.base.medium.clone(),
                ..Default::default()
            }
        ))
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        0.0
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
        self.base.le(ray, lambda)
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        // Do nothing
    }

    fn bounds(&self) -> Option<LightBounds> {
        let p = self.base.render_from_light.apply(Point3f::ZERO);
        let phi = 4.0 * PI * self.scale * self.i.max_value();

        Some(LightBounds::new_with_phi(
            Bounds3f::new(p, p),
            phi,
            Vec3f::new(0.0, 0.0, 1.0),
            -1.0, // cos(pi)
            0.0,  // cos(pi / 2)
            false
        ))
    }
}
