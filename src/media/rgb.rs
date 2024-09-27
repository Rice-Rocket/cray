use std::{collections::HashMap, sync::Arc};

use crate::{color::{sampled::SampledSpectrum, spectrum::{AbstractSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, Spectrum}, wavelengths::SampledWavelengths}, error, phase::{HGPhaseFunction, PhaseFunction}, reader::{paramdict::ParameterDictionary, target::FileLoc}, sampling::SampledGrid, transform::{ApplyInverseTransform, ApplyRayInverseTransform, Transform}, Bounds3f, Float, Point3f, Point3i, Ray};

use super::{iterator::{DDAMajorantIterator, MajorantGrid}, AbstractMedium, MediumProperties};

#[derive(Debug, Clone, PartialEq)]
pub struct RgbGridMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    le_grid: Option<SampledGrid<RgbIlluminantSpectrum>>,
    le_scale: Float,
    phase: HGPhaseFunction,
    sigma_a_grid: Option<SampledGrid<RgbUnboundedSpectrum>>,
    sigma_s_grid: Option<SampledGrid<RgbUnboundedSpectrum>>,
    sigma_scale: Float,
    majorant_grid: MajorantGrid,
}

impl RgbGridMedium {
    pub fn create(
        parameters: &mut ParameterDictionary,
        render_from_medium: Transform,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> RgbGridMedium {
        let sigma_a = parameters.get_rgb_array("sigma_a");
        let sigma_s = parameters.get_rgb_array("sigma_s");
        let le = parameters.get_rgb_array("Le");

        if sigma_a.is_empty() && sigma_s.is_empty() {
            error!(loc, "rgb grid requires 'sigma_a' and/or 'sigma_s' parameter values");
        }

        let n_density = if !sigma_a.is_empty() {
            if !sigma_s.is_empty() && sigma_a.len() != sigma_s.len() {
                error!(loc, "different number of samples ({} vs {}) provided for 'sigma_a' and 'sigma_s'", sigma_a.len(), sigma_s.len());
            }

            sigma_a.len()
        } else {
            sigma_s.len()
        };

        if !le.is_empty() && sigma_a.is_empty() {
            error!(loc, "rgb grid requires 'sigma_a' if 'Le' value provided");
        }

        if !le.is_empty() && n_density != le.len() {
            error!(loc, "expected {} values for 'Le' parameter but was given {}", n_density, le.len());
        }

        let nx = parameters.get_one_int("nx", 1);
        let ny = parameters.get_one_int("ny", 1);
        let nz = parameters.get_one_int("nz", 1);

        if n_density as i32 != nx * ny * nz {
            error!(loc, "rgb grid medium has {} density values; expected nx*ny*nz = {}", n_density, nx * ny * nz);
        }

        let sigma_a_grid = if !sigma_a.is_empty() {
            let color_space = parameters.color_space.as_ref();
            let mut rgb_spectrum_density = Vec::new();

            for rgb in sigma_a {
                rgb_spectrum_density.push(RgbUnboundedSpectrum::new(color_space, &rgb));
            }

            Some(SampledGrid::new(rgb_spectrum_density, nx, ny, nz))
        } else {
            None
        };

        let sigma_s_grid = if !sigma_s.is_empty() {
            let color_space = parameters.color_space.as_ref();
            let mut rgb_spectrum_density = Vec::new();

            for rgb in sigma_s {
                rgb_spectrum_density.push(RgbUnboundedSpectrum::new(color_space, &rgb));
            }

            Some(SampledGrid::new(rgb_spectrum_density, nx, ny, nz))
        } else {
            None
        };

        let le_grid = if !le.is_empty() {
            let color_space = parameters.color_space.as_ref();
            let mut rgb_spectrum_density = Vec::new();

            for rgb in le {
                rgb_spectrum_density.push(RgbIlluminantSpectrum::new(color_space, &rgb));
            }

            Some(SampledGrid::new(rgb_spectrum_density, nx, ny, nz))
        } else {
            None
        };

        let p0 = parameters.get_one_point3f("p0", Point3f::ZERO);
        let p1 = parameters.get_one_point3f("p1", Point3f::ONE);
        let le_scale = parameters.get_one_float("Lescale", 1.0);
        let g = parameters.get_one_float("g", 0.0);
        let sigma_scale = parameters.get_one_float("scale", 1.0);

        RgbGridMedium::new(
            Bounds3f::new(p0, p1),
            render_from_medium,
            g,
            sigma_a_grid,
            sigma_s_grid,
            sigma_scale,
            le_grid,
            le_scale,
        )
    }

    pub fn new(
        bounds: Bounds3f,
        render_from_medium: Transform,
        g: Float,
        sigma_a: Option<SampledGrid<RgbUnboundedSpectrum>>,
        sigma_s: Option<SampledGrid<RgbUnboundedSpectrum>>,
        sigma_scale: Float,
        le: Option<SampledGrid<RgbIlluminantSpectrum>>,
        le_scale: Float,
    ) -> RgbGridMedium {
        if le.is_some() {
            debug_assert!(sigma_a.is_some());
        }

        let mut majorant_grid = MajorantGrid::new(bounds, Point3i::splat(16));

        for z in 0..majorant_grid.res.z {
            for y in 0..majorant_grid.res.y {
                for x in 0..majorant_grid.res.x {
                    let bounds = majorant_grid.voxel_bounds(x, y, z);
                    let max = |s: &RgbUnboundedSpectrum| { s.max_value() };
                    let max_sigma_t = if let Some(ref sig_a) = sigma_a { sig_a.max_value_convert(bounds, max) } else { 1.0 }
                        + if let Some(ref sig_s) = sigma_s { sig_s.max_value_convert(bounds, max) } else { 1.0 };
                    majorant_grid.set(x, y, z, sigma_scale * max_sigma_t);
                }
            }
        }

        RgbGridMedium {
            bounds,
            render_from_medium,
            le_grid: le,
            le_scale,
            phase: HGPhaseFunction::new(g),
            sigma_a_grid: sigma_a,
            sigma_s_grid: sigma_s,
            sigma_scale,
            majorant_grid,
        }
    }
}

impl<'a> AbstractMedium for &'a RgbGridMedium {
    type MajorantIterator = DDAMajorantIterator<'a>;

    fn is_emissive(self) -> bool {
        self.le_grid.is_some() && self.le_scale > 0.0
    }

    fn sample_point(self, mut p: Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        p = self.render_from_medium.apply_inverse(p);
        p = self.bounds.offset(p);

        let convert = |s: &RgbUnboundedSpectrum| { s.sample(lambda) };
        let sigma_a = self.sigma_scale * if let Some(ref sig_a) = self.sigma_a_grid {
            sig_a.lookup_convert(p, convert).unwrap_or(SampledSpectrum::from_const(1.0))
        } else { SampledSpectrum::from_const(1.0) };

        let sigma_s = self.sigma_scale * if let Some(ref sig_s) = self.sigma_s_grid {
            sig_s.lookup_convert(p, convert).unwrap_or(SampledSpectrum::from_const(1.0))
        } else { SampledSpectrum::from_const(1.0) };

        let le = if self.le_grid.is_some() && self.le_scale > 0.0 {
            let convert = |s: &RgbIlluminantSpectrum| { s.sample(lambda) };
            self.le_scale * self.le_grid.as_ref().unwrap().lookup_convert(p, convert).unwrap_or(SampledSpectrum::from_const(0.0))
        } else { SampledSpectrum::from_const(0.0) };

        MediumProperties { sigma_a, sigma_s, phase: PhaseFunction::HenyeyGreenstein(self.phase), le }
    }

    fn sample_ray(self, mut ray: &Ray, mut t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator> {
        let ray = self.render_from_medium.apply_ray_inverse(ray, Some(&mut t_max));
        
        let hits = self.bounds.intersect_p(ray.origin, ray.direction, t_max)?;

        debug_assert!(hits.t1 <= t_max);

        let sigma_t = SampledSpectrum::from_const(1.0);
        Some(DDAMajorantIterator::new(ray, hits.t0, hits.t1, &self.majorant_grid, sigma_t))
    }
}
