use std::{collections::HashMap, sync::Arc};

use tracing::warn;

use crate::{color::{colorspace::RgbColorSpace, sampled::SampledSpectrum, spectrum::{spectrum_to_photometric, AbstractSpectrum, DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, file::resolve_filename, image::Image, media::Medium, options::Options, reader::{paramdict::{ParameterDictionary, SpectrumType}, target::FileLoc}, shape::{AbstractShape, Shape, ShapeSampleContext}, texture::FloatTexture, transform::Transform, Bounds3f, Dot, Float, Normal3f, Point2f, Point3f, Ray, Vec3f, PI};

use super::{AbstractLight, LightBase, LightBounds, LightLiSample, LightSampleContext, LightType};

#[derive(Debug, Clone)]
pub struct DiffuseAreaLight {
    base: LightBase,
    shape: Arc<Shape>,
    // TODO: alpha: FloatTexture,
    area: Float,
    two_sided: bool,
    l_emit: Arc<DenselySampledSpectrum>,
    scale: Float,
    // TODO: image: Image,
    //image color space
}

impl DiffuseAreaLight {
    pub fn new(
        render_from_light: Transform,
        le: Arc<Spectrum>,
        scale: Float,
        shape: Arc<Shape>,
        two_sided: bool,
    ) -> DiffuseAreaLight {
        let area = shape.area();
        let base = LightBase {
            ty: LightType::Area,
            render_from_light,
            medium: None,
        };

        DiffuseAreaLight {
            base,
            shape,
            area,
            two_sided,
            l_emit: Arc::new(DenselySampledSpectrum::new(le.as_ref())),
            scale,
        }
    }

    pub fn create(
        render_from_light: Transform,
        medium: Option<Arc<Medium>>,
        parameters: &mut ParameterDictionary,
        color_space: Arc<RgbColorSpace>,
        loc: &FileLoc,
        shape: Arc<Shape>,
        alpha_tex: Arc<FloatTexture>,
        options: &Options,
    ) -> DiffuseAreaLight {
        // TODO: Use cached_spectra
        let l = parameters.get_one_spectrum(
            "L",
            None,
            SpectrumType::Illuminant,
            &mut HashMap::new(),
        );

        let mut scale = parameters.get_one_float("scale", 1.0);
        let two_sided = parameters.get_one_bool("twosided", false);

        let filename = resolve_filename(
            options,
            &parameters.get_one_string("filename", ""),
        );

        let image: Option<Image> = None;
        if !filename.is_empty() {
            if l.is_some() {
                panic!("Both L and filename specified for diffuse area light");
            }
            todo!("Image area lights not yet implemented");
        }

        let l = if filename.is_empty() && l.is_none() {
            color_space.illuminant.clone()
        } else {
            l.unwrap()
        };

        scale /= spectrum_to_photometric(&l);

        let phi_v = parameters.get_one_float("power", -1.0);
        if phi_v > 0.0 {
            let mut k_e = 1.0;

            if image.is_some() {
                todo!("Image area lights not yet implemented");
            }

            k_e *= if two_sided { 2.0 } else { 1.0 } * shape.area() * PI;
            scale *= phi_v / k_e;
        }

        DiffuseAreaLight::new(render_from_light, l, scale, shape.clone(), two_sided)
    }
}

impl AbstractLight for DiffuseAreaLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        // TODO: Account for image here
        let l = self.l_emit.sample(lambda) * self.scale;
        let double = if self.two_sided { 2.0 } else { 1.0 };
        PI * double * self.area * l
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
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.ns, 0.0);
        let mut ss = self.shape.sample_with_context(&shape_ctx, u)?;

        if ss.pdf == 0.0 || (ss.intr.position() - ctx.p()).length_squared() == 0.0 {
            return None;
        }

        debug_assert!(!ss.pdf.is_nan());

        ss.intr.medium_interface = self.base.medium.clone();

        // TODO: check against the alpha texture with alpha_masked().

        let wi = Vec3f::from((ss.intr.position() - ctx.p()).normalize());
        let le = self.l(ss.intr.position(), ss.intr.n, ss.intr.uv, -wi, lambda);
        if le.is_zero() {
            return None;
        }

        Some(LightLiSample::new(le, wi, ss.pdf, ss.intr))
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.ns, 0.0);
        self.shape.pdf_with_context(&shape_ctx, wi)
    }

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        if !self.two_sided && n.dot(w) < 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        // TODO: check alpha mask with alpha texture and uv.
        // TODO: handle image textures here

        self.scale * self.l_emit.sample(lambda)
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        warn!("le() should only be called for infinite lights");
        SampledSpectrum::from_const(0.0)
    }

    fn preprocess(&mut self, _scene_bounds: &Bounds3f) {
        // Do nothing
    }

    fn bounds(&self) -> Option<LightBounds> {
        let mut phi = 0.0;

        // TODO: handle image textures

        phi = self.l_emit.max_value();
        phi *= self.scale * self.area * PI;

        let nb = self.shape.normal_bounds();
        Some(LightBounds {
            bounds: self.shape.bounds(),
            phi,
            w: nb.w,
            cos_theta_o: nb.cos_theta,
            cos_theta_e: 0.0, // cos(pi / 2)
            two_sided: self.two_sided,
        })
    }
}
