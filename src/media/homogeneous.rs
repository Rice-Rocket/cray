use std::{collections::HashMap, mem::MaybeUninit, sync::Arc};

use tracing::warn;

use crate::{color::{sampled::SampledSpectrum, spectrum::{spectrum_to_photometric, AbstractSpectrum, ConstantSpectrum, DenselySampledSpectrum, Spectrum}, wavelengths::SampledWavelengths}, phase::{HGPhaseFunction, PhaseFunction}, reader::{paramdict::{ParameterDictionary, SpectrumType}, target::FileLoc}, Float, Point3f, Ray};

use super::{iterator::HomogeneousMajorantIterator, preset::get_medium_scattering_properties, AbstractMedium, MediumProperties, RayMajorantSegment};

#[derive(Debug, Clone, PartialEq)]
pub struct HomogeneousMedium {
    sigma_a_spec: DenselySampledSpectrum,
    sigma_s_spec: DenselySampledSpectrum,
    le_spec: DenselySampledSpectrum,
    phase: HGPhaseFunction,
}

impl HomogeneousMedium {
    pub fn create(
        parameters: &mut ParameterDictionary,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> HomogeneousMedium {
        let (mut sig_a, mut sig_s) = (None, None);
        let preset = parameters.get_one_string("preset", "");

        if !preset.is_empty() {
            if let Some((sa, ss)) = get_medium_scattering_properties(&preset) {
                sig_a = Some(sa);
                sig_s = Some(ss);
            } else {
                warn!("{}: material preset {} not found", loc, preset);
            }
        }

        if sig_a.is_none() {
            sig_a = parameters.get_one_spectrum(
                "sigma_a",
                Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)))),
                SpectrumType::Unbounded,
                cached_spectra,
            );

            sig_s = parameters.get_one_spectrum(
                "sigma_s",
                Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)))),
                SpectrumType::Unbounded,
                cached_spectra,
            );
        }

        let mut le = parameters.get_one_spectrum("Le", None, SpectrumType::Illuminant, cached_spectra);
        let mut le_scale = parameters.get_one_float("Lescale", 1.0);
        if le.is_none() || le.as_ref().unwrap().max_value() == 0.0 {
            le = Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.0))));
        } else {
            le_scale /= spectrum_to_photometric(le.as_ref().unwrap());
        }

        let sigma_scale = parameters.get_one_float("scale", 1.0);
        let g = parameters.get_one_float("g", 0.0);

        HomogeneousMedium::new(
            DenselySampledSpectrum::new(&sig_a.unwrap()),
            DenselySampledSpectrum::new(&sig_s.unwrap()),
            sigma_scale,
            DenselySampledSpectrum::new(&le.unwrap()),
            le_scale,
            g,
        )
    }

    pub fn new(
        mut sigma_a: DenselySampledSpectrum,
        mut sigma_s: DenselySampledSpectrum,
        sigma_scale: Float,
        mut le: DenselySampledSpectrum,
        le_scale: Float,
        g: Float
    ) -> HomogeneousMedium {
        sigma_a.scale(sigma_scale);
        sigma_s.scale(sigma_scale);
        le.scale(le_scale);

        HomogeneousMedium {
            sigma_a_spec: sigma_a,
            sigma_s_spec: sigma_s,
            le_spec: le,
            phase: HGPhaseFunction::new(g),
        }
    }
}

impl AbstractMedium for &HomogeneousMedium {
    type MajorantIterator = HomogeneousMajorantIterator;

    fn is_emissive(self) -> bool {
        self.le_spec.max_value() > 0.0
    }

    fn sample_point(self, p: Point3f, lambda: &SampledWavelengths) -> MediumProperties {
        let sigma_a = self.sigma_a_spec.sample(lambda);
        let sigma_s = self.sigma_s_spec.sample(lambda);
        let le = self.le_spec.sample(lambda);

        MediumProperties { sigma_a, sigma_s, phase: PhaseFunction::HenyeyGreenstein(self.phase), le }
    }

    fn sample_ray(self, ray: &Ray, t_max: Float, lambda: &SampledWavelengths) -> Option<Self::MajorantIterator> {
        let sigma_a = self.sigma_a_spec.sample(lambda);
        let sigma_s = self.sigma_s_spec.sample(lambda);

        Some(HomogeneousMajorantIterator::new(0.0, t_max, sigma_a + sigma_s))
    }
}
