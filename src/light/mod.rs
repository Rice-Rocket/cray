use std::{collections::HashMap, io::Write, path::PathBuf, sync::Arc};

use diffuse_area::DiffuseAreaLight;
use image_infinite::ImageInfiniteLight;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use point::PointLight;
use portal_image_infinite::PortalImageInfiniteLight;
use uniform_infinite::UniformInfiniteLight;

use crate::{bounds::Union, camera::CameraTransform, clear_log, color::{colorspace::{NamedColorSpace, RgbColorSpace}, sampled::SampledSpectrum, spectrum::{spectrum_to_photometric, spectrum_to_xyz, DenselySampledSpectrum, RgbIlluminantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, cos_theta, equal_area_square_to_sphere, error, file::resolve_filename, image::{Image, ImageAndMetadata, ImageMetadata, PixelFormat}, interaction::{Interaction, SurfaceInteraction}, log, media::{Medium, MediumInterface}, options::Options, reader::{paramdict::{ParameterDictionary, SpectrumType}, target::FileLoc, utils::truncate_filename}, safe, shape::Shape, sqr, texture::FloatTexture, transform::Transform, warn, Bounds3f, DirectionCone, Dot, Float, Normal3f, Point2f, Point2i, Point3f, Point3fi, Ray, Vec2f, Vec3f, PI};

pub mod sampler;
pub mod point;
pub mod diffuse_area;
pub mod uniform_infinite;
pub mod image_infinite;
pub mod portal_image_infinite;

pub trait AbstractLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum;
    
    fn light_type(&self) -> LightType;

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample>;

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float;

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
    
    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum;

    fn preprocess(&mut self, scene_bounds: &Bounds3f);

    fn bounds(&self) -> Option<LightBounds>;
    
    // TODO: sample_le() and pdf_le() for bidirectional light transport
}

#[derive(Debug, Clone)]
pub enum Light {
    Point(PointLight),
    DiffuseArea(DiffuseAreaLight),
    UniformInfinite(UniformInfiniteLight),
    ImageInfinite(ImageInfiniteLight),
    PortalImageInfinite(PortalImageInfiniteLight),
}

impl Light {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        render_from_light: Transform,
        camera_transform: &CameraTransform,
        outside_medium: Option<Arc<MediumInterface>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) -> Light {
        match name {
            "point" => Light::Point(PointLight::create(
                render_from_light,
                outside_medium,
                parameters,
                parameters.color_space.clone(),
                loc,
                cached_spectra,
            )),
            "infinite" => {
                let color_space = parameters.color_space.clone();
                let l = parameters.get_spectrum_array(
                    "L",
                    SpectrumType::Illuminant,
                    cached_spectra,
                );

                let mut scale = parameters.get_one_float("scale", 1.0);
                let mut portal = parameters.get_point3f_array("portal");
                let filename = resolve_filename(
                    options,
                    parameters.get_one_string("filename", "").as_str(),
                );
                
                let e_v = parameters.get_one_float("illuminance", -1.0);

                if l.is_empty() && filename.is_empty() && portal.is_empty() {
                    scale /= spectrum_to_photometric(&color_space.illuminant);
                    if e_v > 0.0 {
                        let k_e = PI;
                        scale *= e_v / k_e;
                    }

                    Light::UniformInfinite(UniformInfiniteLight::new(
                        render_from_light,
                        color_space.illuminant.clone(),
                        scale,
                    ))
                } else if !l.is_empty() && portal.is_empty() {
                    if !filename.is_empty() {
                        error!(loc, "can't specify both emission L and filename with ImageInfiniteLight");
                    }

                    scale /= spectrum_to_photometric(&l[0]);
                    if e_v > 0.0 {
                        let k_e = PI;
                        scale *= e_v / k_e;
                    }

                    Light::UniformInfinite(UniformInfiniteLight::new(
                        render_from_light,
                        l[0].clone(),
                        scale,
                    ))
                } else {
                    let image_and_metadata = if filename.is_empty() {
                        if let Spectrum::RgbIlluminant(_) = l[0].as_ref() {} else {
                            warn!(loc, "converting non-RGB 'L' parameter to RGB so that image infinite light can be used");
                        };

                        let xyz = spectrum_to_xyz(&l[0]);
                        let rgb = RgbColorSpace::get_named(NamedColorSpace::SRgb).to_rgb(&xyz);

                        let mut im = ImageAndMetadata {
                            image: Image::new(
                                PixelFormat::Float32,
                                Point2i::new(1, 1),
                                &["R".to_string(), "G".to_string(), "B".to_string()],
                                None,
                            ),
                            metadata: ImageMetadata {
                                color_space: Some(RgbColorSpace::get_named(NamedColorSpace::SRgb).clone()),
                                ..Default::default()
                            },
                        };

                        for c in 0..3 {
                            im.image.set_channel(Point2i::new(0, 0), c, rgb[c]);
                        }

                        im
                    } else {
                        let image_and_metadata = Image::read(&PathBuf::from(&filename), None).unwrap();
                        for y in 0..image_and_metadata.image.resolution().y {
                            for x in 0..image_and_metadata.image.resolution().x {
                                for c in 0..image_and_metadata.image.n_channels() {
                                    if image_and_metadata.image.get_channel(Point2i::new(x, y), c).is_nan() {
                                        error!(loc, "image '{}' contains NaN values", truncate_filename(&filename));
                                    }

                                    if image_and_metadata.image.get_channel(Point2i::new(x, y), c).is_infinite() {
                                        error!(loc, "image '{}' contains infinite values", truncate_filename(&filename));
                                    }
                                }
                            }
                        }
                        image_and_metadata
                    };

                    let color_space = image_and_metadata.metadata.color_space.expect("expected color space");
                    let Some(channel_desc) = image_and_metadata.image.get_channel_desc(&["R", "G", "B"]) else {
                        error!(loc, "infinite image light sources must have RGB channels");
                    };

                    scale /= spectrum_to_photometric(&color_space.illuminant);

                    if e_v > 0.0 {
                        let mut illuminance = 0.0;
                        let image = &image_and_metadata.image;
                        let lum = color_space.luminance_vector();

                        log!("Preparing image '{}' for image infinite light...", truncate_filename(&filename));

                        for y in 0..image.resolution().y {
                            let v = (y as Float + 0.5) / image.resolution().y as Float;
                            for x in 0..image.resolution().x {
                                let u = (x as Float + 0.5) / image.resolution().x as Float;
                                let w = equal_area_square_to_sphere(Vec2f::new(u, v));
                                if w.z <= 0.0 {
                                    continue;
                                }

                                let values = image.get_channels(Point2i::new(x, y));
                                for c in 0..3 {
                                    illuminance += values[c] * lum[c] * cos_theta(w);
                                }
                            }
                        }

                        clear_log!();

                        illuminance *= 2.0 * PI / (image.resolution().x * image.resolution().y) as Float;
                        
                        let k_e = illuminance;
                        scale *= e_v / k_e;
                    }


                    log!("Extracting image '{}' for image infinite light...", truncate_filename(&filename));
                    let image = image_and_metadata.image.select_channels(&channel_desc);
                    clear_log!();

                    if !portal.is_empty() {
                        for p in portal.iter_mut() {
                            *p = camera_transform.render_from_world_p(*p);
                        }

                        Light::PortalImageInfinite(PortalImageInfiniteLight::new(
                            render_from_light,
                            Arc::new(image),
                            color_space,
                            scale,
                            &filename,
                            portal,
                            loc,
                        ))
                    } else {
                        Light::ImageInfinite(ImageInfiniteLight::new(
                            render_from_light,
                            Arc::new(image),
                            color_space,
                            scale,
                            &filename,
                            loc,
                        ))
                    }
                }
            },
            _ => { error!(loc, "light '{}' unknown", name); },
        }
    }

    pub fn create_area(
        name: &str,
        parameters: &mut ParameterDictionary,
        render_from_light: Transform,
        medium_interface: Arc<MediumInterface>,
        shape: Arc<Shape>,
        alpha: Arc<FloatTexture>,
        loc: &FileLoc,
        options: &Options,
    ) -> Light {
        match name {
            "diffuse" => Light::DiffuseArea(DiffuseAreaLight::create(
                render_from_light,
                medium_interface.outside.clone(),
                parameters,
                parameters.color_space.clone(),
                loc,
                shape,
                alpha,
                options,
            )),
            _ => { error!(loc, "area light '{}' unknown", name); },
        }
    }
}

impl AbstractLight for Light {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.phi(lambda),
            Light::DiffuseArea(l) => l.phi(lambda),
            Light::UniformInfinite(l) => l.phi(lambda),
            Light::ImageInfinite(l) => l.phi(lambda),
            Light::PortalImageInfinite(l) => l.phi(lambda),
        }
    }

    fn light_type(&self) -> LightType {
        match self {
            Light::Point(l) => l.light_type(),
            Light::DiffuseArea(l) => l.light_type(),
            Light::UniformInfinite(l) => l.light_type(),
            Light::ImageInfinite(l) => l.light_type(),
            Light::PortalImageInfinite(l) => l.light_type(),
        }
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        match self {
            Light::Point(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::DiffuseArea(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::UniformInfinite(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::ImageInfinite(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::PortalImageInfinite(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
        }
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        match self {
            Light::Point(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::DiffuseArea(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::UniformInfinite(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::ImageInfinite(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::PortalImageInfinite(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
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
        match self {
            Light::Point(l) => l.l(p, n, uv, w, lambda),
            Light::DiffuseArea(l) => l.l(p, n, uv, w, lambda),
            Light::UniformInfinite(l) => l.l(p, n, uv, w, lambda),
            Light::ImageInfinite(l) => l.l(p, n, uv, w, lambda),
            Light::PortalImageInfinite(l) => l.l(p, n, uv, w, lambda),
        }
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.le(ray, lambda),
            Light::DiffuseArea(l) => l.le(ray, lambda),
            Light::UniformInfinite(l) => l.le(ray, lambda),
            Light::ImageInfinite(l) => l.le(ray, lambda),
            Light::PortalImageInfinite(l) => l.le(ray, lambda),
        }
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        match self {
            Light::Point(l) => l.preprocess(scene_bounds),
            Light::DiffuseArea(l) => l.preprocess(scene_bounds),
            Light::UniformInfinite(l) => l.preprocess(scene_bounds),
            Light::ImageInfinite(l) => l.preprocess(scene_bounds),
            Light::PortalImageInfinite(l) => l.preprocess(scene_bounds),
        }
    }

    fn bounds(&self) -> Option<LightBounds> {
        match self {
            Light::Point(l) => l.bounds(),
            Light::DiffuseArea(l) => l.bounds(),
            Light::UniformInfinite(l) => l.bounds(),
            Light::ImageInfinite(l) => l.bounds(),
            Light::PortalImageInfinite(l) => l.bounds(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LightBase {
    pub(super) ty: LightType,
    pub(super) render_from_light: Transform,
    pub(super) medium: Option<Arc<MediumInterface>>,
}

impl LightBase {
    pub fn light_type(&self) -> LightType {
        self.ty
    }

    pub fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    // TODO: Implement caching system for DenselySampledSpectrum
    // Include lookup_spectrum() to get cached DenselySampledSpectrum
}

#[derive(Debug, Clone, Default)]
pub struct LightSampleContext {
    pub pi: Point3fi,
    pub n: Normal3f,
    pub ns: Normal3f,
}

impl LightSampleContext {
    pub fn new(pi: Point3fi, n: Normal3f, ns: Normal3f) -> LightSampleContext {
        LightSampleContext { pi, n, ns }
    }

    pub fn p(&self) -> Point3f {
        Point3f::from(self.pi)
    }
}

impl From<&SurfaceInteraction> for LightSampleContext {
    fn from(value: &SurfaceInteraction) -> Self {
        LightSampleContext {
            pi: value.interaction.pi,
            n: value.interaction.n,
            ns: value.shading.n,
        }
    }
}

impl From<&Interaction> for LightSampleContext {
    fn from(value: &Interaction) -> Self {
        LightSampleContext {
            pi: value.pi,
            n: Normal3f::ZERO,
            ns: Normal3f::ZERO,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LightLiSample {
    pub l: SampledSpectrum,
    pub wi: Vec3f,
    pub pdf: Float,
    pub p_light: Interaction,
}

impl LightLiSample {
    pub fn new(
        l: SampledSpectrum,
        wi: Vec3f,
        pdf: Float,
        p_light: Interaction,
    ) -> LightLiSample {
        LightLiSample {
            l,
            wi,
            pdf,
            p_light,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LightType {
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite,
}

impl LightType {
    pub fn is_delta(&self) -> bool {
        *self == Self::DeltaDirection || *self == Self::DeltaPosition
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LightBounds {
    pub bounds: Bounds3f,
    pub phi: Float,
    pub w: Vec3f,
    pub cos_theta_o: Float,
    pub cos_theta_e: Float,
    pub two_sided: bool,
}

impl LightBounds {
    pub fn new(
        bounds: Bounds3f,
        w: Vec3f,
        cos_theta_o: Float,
        cos_theta_e: Float,
        two_sided: bool,
    ) -> LightBounds {
        LightBounds {
            bounds,
            phi: 0.0,
            w,
            cos_theta_o,
            cos_theta_e,
            two_sided,
        }
    }

    pub fn new_with_phi(
        bounds: Bounds3f,
        phi: Float,
        w: Vec3f,
        cos_theta_o: Float,
        cos_theta_e: Float,
        two_sided: bool,
    ) -> LightBounds {
        LightBounds {
            bounds,
            phi,
            w,
            cos_theta_o,
            cos_theta_e,
            two_sided,
        }
    }

    pub fn union(&self, other: &LightBounds) -> LightBounds {
        if self.phi == 0.0 {
            return *other;
        }
        if other.phi == 0.0 {
            return *self;
        }

        let cone = DirectionCone::new(self.w, self.cos_theta_o).union(DirectionCone::new(other.w, other.cos_theta_o));
        let cos_theta_o = cone.cos_theta;
        let cos_theta_e = self.cos_theta_e.min(other.cos_theta_e);

        LightBounds {
            bounds: self.bounds.union(other.bounds),
            phi: self.phi + other.phi,
            w: cone.w,
            cos_theta_o,
            cos_theta_e,
            two_sided: self.two_sided || other.two_sided,
        }
    }

    pub fn centroid(&self) -> Point3f {
        (self.bounds.min + self.bounds.max) / 2.0
    }

    pub fn importance(&self, p: Point3f, n: Normal3f) -> Float {
        let pc = (self.bounds.min + self.bounds.max) / 2.0;
        let mut d2 = p.distance_squared(pc);
        d2 = Float::max(d2, self.bounds.diagonal().length() / 2.0);

        let cos_sub_clamped = |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| -> Float {
            if cos_theta_a > cos_theta_b {
                return 1.0;
            }

            cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b
        };

        let sin_sub_clamped = |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| -> Float {
            if cos_theta_a > cos_theta_b {
                return 0.0;
            }

            sin_theta_a * cos_theta_b - cos_theta_a * sin_theta_b
        };

        let wi = (p - pc).normalize();
        let mut cos_theta_w = self.w.dot(wi);
        if self.two_sided {
            cos_theta_w = cos_theta_w.abs();
        }
        let sin_theta_w = safe::sqrt(1.0 - sqr(cos_theta_w));

        let cos_theta_b = DirectionCone::bound_subtended_directions(self.bounds, p).cos_theta;
        let sin_theta_b = safe::sqrt(1.0 - sqr(cos_theta_b));

        let sin_theta_o = safe::sqrt(1.0 - sqr(self.cos_theta_o));
        let cos_theta_x = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let sin_theta_x = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);

        if cos_theta_p <= self.cos_theta_e {
            return 0.0;
        }

        let mut importance = self.phi * cos_theta_p / d2;
        debug_assert!(importance >= -1e-3);

        if n != Normal3f::ZERO {
            let cos_theta_i = wi.dot(n).abs();
            let sin_theta_i = safe::sqrt(1.0 - sqr(cos_theta_i));
            let cos_theta_pi = cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
            importance *= cos_theta_pi;
        }
        
        importance = Float::max(importance, 0.0);
        importance
    }
}

impl Default for LightBounds {
    fn default() -> Self {
        Self {
            bounds: Default::default(),
            phi: 0.0,
            w: Default::default(),
            cos_theta_o: Default::default(),
            cos_theta_e: Default::default(),
            two_sided: Default::default(),
        }
    }
}
