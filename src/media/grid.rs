use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use tracing::error;

use crate::{color::{sampled::SampledSpectrum, spectrum::{spectrum_to_photometric, AbstractSpectrum, BlackbodySpectrum, ConstantSpectrum, DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, phase::{HGPhaseFunction, PhaseFunction}, reader::{paramdict::{ParameterDictionary, SpectrumType}, target::FileLoc}, sampling::SampledGrid, transform::{ApplyInverseTransform, ApplyRayInverseTransform, Transform}, Bounds3f, Float, Point3f, Point3i, Ray};

use super::{iterator::{DDAMajorantIterator, MajorantGrid}, AbstractMedium, MediumProperties};

#[derive(Debug, Clone, PartialEq)]
pub struct GridMedium {
    bounds: Bounds3f,
    render_from_medium: Transform,
    sigma_a_spec: DenselySampledSpectrum,
    sigma_s_spec: DenselySampledSpectrum,
    density_grid: SampledGrid<Float>,
    phase: HGPhaseFunction,
    temperature_grid: Option<SampledGrid<Float>>,
    le_spec: DenselySampledSpectrum,
    le_scale: SampledGrid<Float>,
    is_emissive: bool,
    temperature_scale: Float,
    temperature_offset: Float,
    majorant_grid: MajorantGrid,
}

impl GridMedium {
    pub fn create(
        parameters: &mut ParameterDictionary,
        render_from_medium: Transform,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> GridMedium {
        let density = parameters.get_float_array("density");
        let temperature = parameters.get_float_array("temperature");

        if density.is_empty() {
            error!("{}: No 'density' value provided for grid medium", loc);
        }

        let n_density = density.len();

        if !temperature.is_empty() && n_density != temperature.len() {
            error!("{}: Different number of samples ({} vs {}) provided for 'density' and 'temperature'", loc, n_density, temperature.len());
        }

        let nx = parameters.get_one_int("nx", 1);
        let ny = parameters.get_one_int("ny", 1);
        let nz = parameters.get_one_int("nz", 1);

        if n_density as i32 != nx * ny * nz {
            error!("{}: Grid medium has {} density values; expected nx*ny*nz = {}", loc, n_density, nx * ny * nz);
        }

        let density_grid = SampledGrid::new(density, nx, ny, nz);
        let temperature_grid = if !temperature.is_empty() {
            Some(SampledGrid::new(temperature, nx, ny, nz))
        } else {
            None
        };

        let mut le = parameters.get_one_spectrum("Le", None, SpectrumType::Illuminant, cached_spectra);

        if le.is_some() && temperature_grid.is_some() {
            error!("{}: Both 'Le' and 'temperature' values were provided", loc);
        }

        let mut le_norm = 1.0;
        if le.is_none() || le.as_ref().unwrap().max_value() == 0.0 {
            le = Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.0))))
        } else {
            le_norm = 1.0 / spectrum_to_photometric(le.as_ref().unwrap());
        };

        let mut le_scale = parameters.get_float_array("Lescale");
        let le_grid = if le_scale.is_empty() {
            SampledGrid::new(vec![le_norm], 1, 1, 1)
        } else {
            if le_scale.len() as i32 != nx * ny * nz {
                error!("{}: Expected {}*{}*{} = {} values for 'Lescale' but was given {}", loc, nx, ny, nz, nx * ny * nz, le_scale.len());
            }

            for scale in le_scale.iter_mut() {
                *scale *= le_norm;
            }
            SampledGrid::new(le_scale, nx, ny, nz)
        };

        let p0 = parameters.get_one_point3f("p0", Point3f::new(0.0, 0.0, 0.0));
        let p1 = parameters.get_one_point3f("p1", Point3f::new(1.0, 1.0, 1.0));

        let g = parameters.get_one_float("g", 0.0);

        let sigma_a = parameters.get_one_spectrum(
            "sigma_a",
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)))),
            SpectrumType::Unbounded,
            cached_spectra,
        );

        let sigma_s = parameters.get_one_spectrum(
            "sigma_s",
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)))),
            SpectrumType::Unbounded,
            cached_spectra,
        );

        let sigma_scale = parameters.get_one_float("scale", 1.0);
        let temperature_cutoff = parameters.get_one_float("temperaturecutoff", 0.0);
        let temperature_offset = parameters.get_one_float("temperatureoffset", temperature_cutoff);
        let temperature_scale = parameters.get_one_float("temperaturescale", 1.0);

        GridMedium::new(
            Bounds3f::new(p0, p1),
            render_from_medium,
            DenselySampledSpectrum::new(sigma_a.unwrap().as_ref()),
            DenselySampledSpectrum::new(sigma_s.unwrap().as_ref()),
            sigma_scale,
            g,
            density_grid,
            temperature_grid,
            temperature_scale,
            temperature_offset,
            DenselySampledSpectrum::new(le.unwrap().as_ref()),
            le_grid,
        )
    }

    pub fn new(
        bounds: Bounds3f,
        render_from_medium: Transform,
        mut sigma_a: DenselySampledSpectrum,
        mut sigma_s: DenselySampledSpectrum,
        sigma_scale: Float,
        g: Float,
        density: SampledGrid<Float>,
        temperature: Option<SampledGrid<Float>>,
        temperature_scale: Float,
        temperature_offset: Float,
        le: DenselySampledSpectrum,
        le_scale: SampledGrid<Float>,
    ) -> GridMedium {
        sigma_a.scale(sigma_scale);
        sigma_s.scale(sigma_scale);

        let is_emissive = if temperature.is_some() { true } else { le.max_value() > 0.0 };

        let mut majorant_grid = MajorantGrid::new(bounds, Point3i::splat(16));
        for z in 0..majorant_grid.res.z {
            for y in 0..majorant_grid.res.y {
                for x in 0..majorant_grid.res.x {
                    let vbounds = majorant_grid.voxel_bounds(x, y, z);
                    majorant_grid.set(x, y, z, density.max_value(bounds));
                }
            }
        }

        GridMedium {
            bounds,
            render_from_medium,
            sigma_a_spec: sigma_a,
            sigma_s_spec: sigma_s,
            density_grid: density,
            phase: HGPhaseFunction::new(g),
            temperature_grid: temperature,
            le_spec: le,
            le_scale,
            is_emissive,
            majorant_grid,
            temperature_scale,
            temperature_offset,
        }
    }
}

impl<'a> AbstractMedium for &'a GridMedium {
    type MajorantIterator = DDAMajorantIterator<'a>;

    fn is_emissive(self) -> bool {
        self.is_emissive
    }

    fn sample_point(self, mut p: Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let mut sigma_a = self.sigma_a_spec.sample(lambda);
        let mut sigma_s = self.sigma_s_spec.sample(lambda);

        p = self.render_from_medium.apply_inverse(p);
        p = self.bounds.offset(p);

        let d = self.density_grid.lookup(p).unwrap_or(0.0);
        sigma_a = sigma_a * d;
        sigma_s = sigma_s * d;

        let mut le = SampledSpectrum::from_const(0.0);

        if self.is_emissive {
            let scale = self.le_scale.lookup(p).unwrap_or(0.0);
            if scale > 0.0 {
                if let Some(ref grid) = self.temperature_grid {
                    let mut temp = grid.lookup(p).unwrap_or(0.0);
                    temp = (temp - self.temperature_offset) * self.temperature_scale;
                    if temp > 100.0 {
                        le = scale * BlackbodySpectrum::new(temp).sample(lambda);
                    }
                } else {
                    le = scale * self.le_spec.sample(lambda);
                }
            }
        }

        MediumProperties { sigma_a, sigma_s, phase: PhaseFunction::HenyeyGreenstein(self.phase), le }
    }

    fn sample_ray(self, mut ray: &Ray, mut t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator> {
        let ray = self.render_from_medium.apply_ray_inverse(ray, Some(&mut t_max));

        let hits = self.bounds.intersect_p(ray.origin, ray.direction, t_max)?;

        debug_assert!(hits.t1 <= t_max);

        let sigma_a = self.sigma_a_spec.sample(lambda);
        let sigma_s = self.sigma_s_spec.sample(lambda);
        let sigma_t = sigma_a + sigma_s;

        Some(DDAMajorantIterator::new(ray, hits.t0, hits.t1, &self.majorant_grid, sigma_t))
    }
}
