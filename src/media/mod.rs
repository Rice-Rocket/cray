use std::{collections::HashMap, sync::Arc};

use grid::GridMedium;
use homogeneous::HomogeneousMedium;
use iterator::{HomogeneousMajorantIterator, RayMajorantIterator};
use rand::{rngs::SmallRng, Rng};
use rgb::RgbGridMedium;

use crate::{color::{sampled::SampledSpectrum, spectrum::Spectrum, wavelengths::SampledWavelengths}, phase::PhaseFunction, ray::AbstractRay, reader::{paramdict::ParameterDictionary, target::FileLoc}, sampling::sample_exponential, transform::Transform, Float, Point3f, Ray};

pub mod iterator;
pub mod homogeneous;
pub mod grid;
pub mod rgb;
pub mod preset;

pub trait AbstractMedium {
    type MajorantIterator: Iterator<Item = RayMajorantSegment>;

    #[allow(clippy::wrong_self_convention)]
    fn is_emissive(self) -> bool;

    fn sample_point(self, p: Point3f, lambda: &SampledWavelengths) -> MediumProperties;

    fn sample_ray(self, ray: &Ray, t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator>;
}

// TODO: Test if boxing here is faster
#[derive(Debug, Clone, PartialEq)]
pub enum Medium {
    Homogeneous(Box<HomogeneousMedium>),
    Grid(Box<GridMedium>),
    RgbGrid(Box<RgbGridMedium>),
}

impl Medium {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        render_from_medium: Transform,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> Medium {
        match name {
            "homogeneous" => Medium::Homogeneous(Box::new(HomogeneousMedium::create(parameters, cached_spectra, loc))),
            "uniformgrid" => Medium::Grid(Box::new(GridMedium::create(parameters, render_from_medium, cached_spectra, loc))),
            "rgbgrid" => Medium::RgbGrid(Box::new(RgbGridMedium::create(parameters, render_from_medium, cached_spectra, loc))),
            _ => panic!("{}: Unknown medium {}", loc, name),
        }
    }
}

impl<'a> AbstractMedium for &'a Medium {
    type MajorantIterator = RayMajorantIterator<'a>;

    fn is_emissive(self) -> bool {
        match self {
            Medium::Homogeneous(m) => m.is_emissive(),
            Medium::Grid(m) => m.is_emissive(),
            Medium::RgbGrid(m) => m.is_emissive(),
        }
    }

    fn sample_point(self, p: Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        match self {
            Medium::Homogeneous(m) => m.sample_point(p, lambda),
            Medium::Grid(m) => m.sample_point(p, lambda),
            Medium::RgbGrid(m) => m.sample_point(p, lambda),
        }
    }

    fn sample_ray(self, ray: &Ray, t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator> {
        match self {
            Medium::Homogeneous(m) => Some(RayMajorantIterator::Homogeneous(m.sample_ray(ray, t_max, lambda)?)),
            Medium::Grid(m) => Some(RayMajorantIterator::DDA(m.sample_ray(ray, t_max, lambda)?)),
            Medium::RgbGrid(m) => Some(RayMajorantIterator::DDA(m.sample_ray(ray, t_max, lambda)?)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MediumInterface {
    pub inside: Option<Arc<Medium>>,
    pub outside: Option<Arc<Medium>>,
}

impl MediumInterface {
    pub fn new(inside: Option<Arc<Medium>>, outside: Option<Arc<Medium>>) -> MediumInterface {
        MediumInterface { inside, outside }
    }

    pub fn empty() -> MediumInterface {
        MediumInterface::new(None, None)
    }

    pub fn is_some(&self) -> bool {
        self.inside.is_some() || self.outside.is_some()
    }

    pub fn is_transition(&self) -> bool {
        self.inside != self.outside
    }
}

impl From<Option<MediumInterface>> for MediumInterface {
    fn from(value: Option<MediumInterface>) -> Self {
        match value {
            Some(mi) => mi,
            None => MediumInterface::empty(),
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

pub fn sample_t_maj<F>(
    ray: &mut Ray,
    mut t_max: Float,
    mut u: Float,
    rng: &mut SmallRng,
    lambda: &SampledWavelengths,
    mut callback: F,
) -> SampledSpectrum
where
    F: FnMut(&mut Ray, Point3f, &MediumProperties, &SampledSpectrum, &SampledSpectrum, &mut SmallRng) -> bool,
{
    let rd_length = ray.direction.length();
    t_max *= rd_length;
    ray.direction /= rd_length;

    let Some(medium) = ray.medium.clone() else {
        return SampledSpectrum::from_const(0.0);
    };

    let Some(mut iter) = medium.sample_ray(ray, t_max, lambda) else {
        return SampledSpectrum::from_const(0.0);
    };

    let mut t_maj = SampledSpectrum::from_const(1.0);
    let mut done = false;

    while !done {
        let Some(seg) = iter.next() else {
            return t_maj;
        };

        if seg.sigma_maj[0] == 0.0 {
            let mut dt = seg.t_max - seg.t_min;

            if dt.is_infinite() {
                dt = Float::MAX;
            }

            // TODO: fast_exp()
            t_maj *= (-dt * seg.sigma_maj).exp();
            continue;
        }

        let mut t_min = seg.t_min;
        loop {
            let t = t_min + sample_exponential(u, seg.sigma_maj[0]);
            u = rng.gen();

            if t < seg.t_max {
                // TODO: fast_exp()
                t_maj *= (-(t - t_min) * seg.sigma_maj).exp();
                let mp = medium.sample_point(ray.at(t), lambda);

                if !callback(ray, ray.at(t), &mp, &seg.sigma_maj, &t_maj, rng) {
                    done = true;
                    break;
                }

                t_maj = SampledSpectrum::from_const(1.0);
                t_min = t;
            } else {
                let mut dt = seg.t_max - t_min;

                if dt.is_infinite() {
                    dt = Float::MAX;
                }

                // TODO: fast_exp()
                t_maj *= (-dt * seg.sigma_maj).exp();
                break;
            }
        }
    }

    SampledSpectrum::from_const(1.0)
}
