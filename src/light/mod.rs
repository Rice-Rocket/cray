use std::{collections::HashMap, sync::Arc};

use point::PointLight;

use crate::{camera::CameraTransform, color::{sampled::SampledSpectrum, spectrum::{DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, interaction::{Interaction, SurfaceInteraction}, media::{Medium, MediumInterface}, options::Options, reader::{paramdict::ParameterDictionary, target::FileLoc}, transform::Transform, Bounds3f, DirectionCone, Float, Normal3f, Point2f, Point3f, Point3fi, Ray, Vec3f};

pub mod point;

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
            _ => panic!("{}: Light {} unknown", loc, name)
        }
    }
}

impl AbstractLight for Light {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.phi(lambda),
        }
    }

    fn light_type(&self) -> LightType {
        match self {
            Light::Point(l) => l.light_type(),
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
        }
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        match self {
            Light::Point(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
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
        }
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.le(ray, lambda),
        }
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        match self {
            Light::Point(l) => l.preprocess(scene_bounds),
        }
    }

    fn bounds(&self) -> Option<LightBounds> {
        match self {
            Light::Point(l) => l.bounds(),
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

impl From<SurfaceInteraction> for LightSampleContext {
    fn from(value: SurfaceInteraction) -> Self {
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

#[derive(Debug, Clone)]
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
            return other.clone();
        }
        if other.phi == 0.0 {
            return self.clone();
        }

        let cone = DirectionCone::new(self.w, self.cos_theta_o).union(DirectionCone::new(other.w, other.cos_theta_o));
        let cos_theta_o = cone.cos_theta;
        let cos_theta_e = self.cos_theta_e.min(other.cos_theta_e);

        LightBounds {
            bounds: self.bounds.union_box(other.bounds),
            phi: self.phi + other.phi,
            w: cone.w,
            cos_theta_o,
            cos_theta_e,
            two_sided: self.two_sided || other.two_sided,
        }
    }
}
