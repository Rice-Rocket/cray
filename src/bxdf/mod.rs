use diffuse::DiffuseBxDF;

use crate::{abs_cos_theta, color::sampled::SampledSpectrum, sampling::{sample_uniform_hemisphere, uniform_hemisphere_pdf}, Float, Point2f, Vec3f, PI};

pub mod diffuse;

pub trait AbstractBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum;

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample>;

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float;

    fn flags(&self) -> BxDFFlags;

    fn rho_hd(&self, wo: Vec3f, uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        if wo.z == 0.0 {
            return SampledSpectrum::default();
        }

        let mut r = SampledSpectrum::from_const(0.0);
        debug_assert_eq!(uc.len(), u2.len());

        for i in 0..uc.len() {
            let bs = self.sample_f(
                wo,
                uc[i],
                u2[i],
                TransportMode::Radiance,
                BxDFReflTransFlags::ALL,
            );

            if let Some(bs) = bs {
                if bs.pdf > 0.0 {
                    r += bs.f * abs_cos_theta(bs.wi) / bs.pdf;
                }
            }
        }

        r / uc.len() as Float
    }

    fn rho_hh(&self, u1: &[Point2f], uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        debug_assert_eq!(u1.len(), uc.len());
        debug_assert_eq!(uc.len(), u2.len());

        let mut r = SampledSpectrum::from_const(0.0);
        for i in 0..uc.len() {
            let wo = sample_uniform_hemisphere(u1[i]);
            if wo.z == 0.0 {
                continue;
            }

            let pdfo = uniform_hemisphere_pdf();
            let bs = self.sample_f(
                wo,
                uc[i],
                u2[i],
                TransportMode::Radiance,
                BxDFReflTransFlags::ALL,
            );

            if let Some(bs) = bs {
                if bs.pdf > 0.0 {
                    r += bs.f * abs_cos_theta(bs.wi) * abs_cos_theta(wo) / (pdfo * bs.pdf);
                }
            }
        }

        r / (PI * uc.len() as Float)
    }

    fn regularize(&mut self);
}

pub enum BxDF {
    Diffuse(DiffuseBxDF),
}

impl AbstractBxDF for BxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        match self {
            BxDF::Diffuse(v) => v.f(wo, wi, mode),
        }
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        match self {
            BxDF::Diffuse(v) => v.sample_f(wo, uc, u, mode, sample_flags),
        }
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        match self {
            BxDF::Diffuse(v) => v.pdf(wo, wi, mode, sample_flags),
        }
    }

    fn flags(&self) -> BxDFFlags {
        match self {
            BxDF::Diffuse(v) => v.flags(),
        }
    }

    fn regularize(&mut self) {
        match self {
            BxDF::Diffuse(v) => v.regularize(),
        }
    }
}

pub struct BSDFSample {
    pub f: SampledSpectrum,
    pub wi: Vec3f,
    pub pdf: Float,
    pub flags: BxDFFlags,
    pub eta: Float,
    pub pdf_is_proportional: bool,
}

impl BSDFSample {
    pub fn new(f: SampledSpectrum, wi: Vec3f, pdf: Float, flags: BxDFFlags) -> BSDFSample {
        BSDFSample {
            f,
            wi,
            pdf,
            flags,
            eta: 1.0,
            pdf_is_proportional: false,
        }
    }

    pub fn new_with_eta(f: SampledSpectrum, wi: Vec3f, pdf: Float, flags: BxDFFlags, eta: Float) -> BSDFSample {
        BSDFSample {
            f,
            wi,
            pdf,
            flags,
            eta,
            pdf_is_proportional: false,
        }
    }

    pub fn is_reflection(&self) -> bool {
        self.flags.is_reflective()
    }

    pub fn is_transmission(&self) -> bool {
        self.flags.is_transmissive()
    }

    pub fn is_diffuse(&self) -> bool {
        self.flags.is_diffuse()
    }

    pub fn is_glossy(&self) -> bool {
        self.flags.is_glossy()
    }

    pub fn is_specular(&self) -> bool {
        self.flags.is_specular()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TransportMode {
    Radiance,
    Importance,
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BxDFReflTransFlags: u8 {
        const UNSET = 0;
        const REFLECTION = 1 << 0;
        const TRANSMISSION = 1 << 1;
        const ALL = Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }
}

bitflags::bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BxDFFlags: u8 {
        const UNSET = 0;
        const REFLECTION = 1 << 0;
        const TRANSMISSION = 1 << 1;
        const DIFFUSE = 1 << 2;
        const GLOSSY = 1 << 3;
        const SPECULAR = 1 << 4;
        const DIFFUSE_REFLECTION = Self::DIFFUSE.bits() | Self::REFLECTION.bits();
        const DIFFUSE_TRANSMISSION = Self::DIFFUSE.bits() | Self::TRANSMISSION.bits();
        const GLOSSY_REFLECTION = Self::GLOSSY.bits() | Self::REFLECTION.bits();
        const GLOSSY_TRANSMISSION = Self::GLOSSY.bits() | Self::TRANSMISSION.bits();
        const SPECULAR_REFLECTION = Self::SPECULAR.bits() | Self::REFLECTION.bits();
        const SPECULAR_TRANSMISSION = Self::SPECULAR.bits() | Self::TRANSMISSION.bits();
        const ALL = Self::DIFFUSE.bits() | Self::SPECULAR.bits() | Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }
}

impl BxDFFlags {
    pub fn is_reflective(&self) -> bool {
        (*self & Self::REFLECTION).bits() != 0
    }

    pub fn is_transmissive(&self) -> bool {
        (*self & Self::TRANSMISSION).bits() != 0
    }

    pub fn is_diffuse(&self) -> bool {
        (*self & Self::DIFFUSE).bits() != 0
    }

    pub fn is_glossy(&self) -> bool {
        (*self & Self::GLOSSY).bits() != 0
    }

    pub fn is_specular(&self) -> bool {
        (*self & Self::SPECULAR).bits() != 0
    }

    pub fn is_non_specular(&self) -> bool {
        (*self & (Self::DIFFUSE | Self::GLOSSY)).bits() != 0
    }
}
