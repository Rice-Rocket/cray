use grid::GridMedium;
use homogeneous::HomogeneousMedium;
use iterator::{HomogeneousMajorantIterator, RayMajorantIterator};

use crate::{color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, phase::PhaseFunction, Float, Point3f, Ray};

pub mod iterator;
pub mod homogeneous;
pub mod grid;
pub mod rgb;
pub mod preset;

pub trait AbstractMedium {
    type MajorantIterator: Iterator<Item = RayMajorantSegment>;

    fn is_emissive(self) -> bool;

    fn sample_point(self, p: Point3f, lambda: &SampledWavelengths) -> MediumProperties;

    fn sample_ray(self, ray: Ray, t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator>;
}

#[derive(Debug, Clone)]
pub enum Medium {
    Homogeneous(HomogeneousMedium),
    Grid(GridMedium),
}

impl<'a> AbstractMedium for &'a Medium {
    type MajorantIterator = RayMajorantIterator<'a>;

    fn is_emissive(self) -> bool {
        match self {
            Medium::Homogeneous(m) => m.is_emissive(),
            Medium::Grid(m) => m.is_emissive(),
        }
    }

    fn sample_point(self, p: Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        match self {
            Medium::Homogeneous(m) => m.sample_point(p, lambda),
            Medium::Grid(m) => m.sample_point(p, lambda),
        }
    }

    fn sample_ray(self, ray: Ray, t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator> {
        match self {
            Medium::Homogeneous(m) => Some(RayMajorantIterator::Homogeneous(m.sample_ray(ray, t_max, lambda)?)),
            Medium::Grid(m) => Some(RayMajorantIterator::DDA(m.sample_ray(ray, t_max, lambda)?)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MediumProperties {
    pub sigma_a: SampledSpectrum,
    pub sigma_s: SampledSpectrum,
    pub phase: PhaseFunction,
    pub le: SampledSpectrum,
}

#[derive(Debug, Clone, Copy)]
pub struct RayMajorantSegment {
    pub t_min: Float,
    pub t_max: Float,
    pub sigma_maj: SampledSpectrum,
}
