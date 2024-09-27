use std::{collections::HashMap, sync::Arc};

use once_cell::sync::Lazy;

use crate::{color::cie::CIE_Y_INTEGRAL, file::read_float_file, math::*, warn};

use super::{cie::{Cie, NUM_CIES_SAMPLES}, colorspace::RgbColorSpace, named_spectrum::{NamedSpectrum, CIE_S0, CIE_S1, CIE_S2, CIE_S_LAMBDA}, rgb_xyz::{Rgb, RgbSigmoidPolynomial, Xyz}, sampled::{SampledSpectrum, NUM_SPECTRUM_SAMPLES}, wavelengths::SampledWavelengths};

/// Minimum wavelength of visible light. Nanometers
pub const LAMBDA_MIN: Float = 360.0;
/// Maximum wavelength of visible light. Nanometers
pub const LAMBDA_MAX: Float = 830.0;

pub trait AbstractSpectrum {
    fn get(&self, lambda: Float) -> Float;
    fn max_value(&self) -> Float;
    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug, PartialEq, Clone)]
pub enum Spectrum {
    Constant(ConstantSpectrum),
    DenselySampled(DenselySampledSpectrum),
    PiecewiseLinear(PiecewiseLinearSpectrum),
    Blackbody(BlackbodySpectrum),
    RgbAlbedo(RgbAlbedoSpectrum),
    RgbUnbounded(RgbUnboundedSpectrum),
    RgbIlluminant(RgbIlluminantSpectrum),
}

impl Spectrum {
    pub fn read(filename: &str, cached_spectra: &mut HashMap<String, Arc<Spectrum>>) -> Option<Arc<Spectrum>> {
        let spectrum = cached_spectra.get(filename);
        if let Some(spectrum) = spectrum {
            return Some(spectrum.clone());
        }

        let pls = PiecewiseLinearSpectrum::read(filename).map(Spectrum::PiecewiseLinear);
        if let Some(pls) = pls {
            let spectrum = Arc::new(pls);
            cached_spectra.insert(filename.to_string(), spectrum.clone());
            Some(spectrum)
        } else {
            None
        }
    }

    pub fn get_named_spectrum(spectrum: NamedSpectrum) -> Arc<Spectrum> {
        use super::named_spectrum::*;

        match spectrum {
            NamedSpectrum::StdIllumD65 => Lazy::force(&STD_ILLUM_D65).clone(),
            NamedSpectrum::IllumAcesD60 => Lazy::force(&ILLUM_ACES_D60).clone(),
            NamedSpectrum::GlassBk7 => Lazy::force(&GLASS_BK7_ETA).clone(),
            NamedSpectrum::GlassF11 => Lazy::force(&GLASS_F11_ETA).clone(),
            NamedSpectrum::GlassBaf10 => Lazy::force(&GLASS_BAF10_ETA).clone(),
            NamedSpectrum::CuEta => Lazy::force(&CU_ETA).clone(),
            NamedSpectrum::CuK => Lazy::force(&CU_K).clone(),
            NamedSpectrum::AuEta => Lazy::force(&AU_ETA).clone(),
            NamedSpectrum::AuK => Lazy::force(&AU_K).clone(),
            NamedSpectrum::AgEta => Lazy::force(&AG_ETA).clone(),
            NamedSpectrum::AgK => Lazy::force(&AG_K).clone(),
            NamedSpectrum::AlEta => Lazy::force(&AL_ETA).clone(),
            NamedSpectrum::AlK => Lazy::force(&AL_K).clone(),
        }
    }

    pub fn get_cie(cie: Cie) -> &'static Spectrum {
        match cie {
            Cie::X => Lazy::force(&super::cie::X),
            Cie::Y => Lazy::force(&super::cie::Y),
            Cie::Z => Lazy::force(&super::cie::Z),
        }
    }
}

impl AbstractSpectrum for Spectrum {
    /// Gets the value of the spectral distributions at wavelength `lambda`.
    fn get(&self, lambda: Float) -> Float {
        match self {
            Spectrum::Constant(s) => s.get(lambda),
            Spectrum::DenselySampled(s) => s.get(lambda),
            Spectrum::PiecewiseLinear(s) => s.get(lambda),
            Spectrum::Blackbody(s) => s.get(lambda),
            Spectrum::RgbAlbedo(s) => s.get(lambda),
            Spectrum::RgbUnbounded(s) => s.get(lambda),
            Spectrum::RgbIlluminant(s) => s.get(lambda),
        }
    }

    fn max_value(&self) -> Float {
        match self {
            Spectrum::Constant(s) => s.max_value(),
            Spectrum::DenselySampled(s) => s.max_value(),
            Spectrum::PiecewiseLinear(s) => s.max_value(),
            Spectrum::Blackbody(s) => s.max_value(),
            Spectrum::RgbAlbedo(s) => s.max_value(),
            Spectrum::RgbUnbounded(s) => s.max_value(),
            Spectrum::RgbIlluminant(s) => s.max_value(),
        }
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Spectrum::Constant(s) => s.sample(lambda),
            Spectrum::DenselySampled(s) => s.sample(lambda),
            Spectrum::PiecewiseLinear(s) => s.sample(lambda),
            Spectrum::Blackbody(s) => s.sample(lambda),
            Spectrum::RgbAlbedo(s) => s.sample(lambda),
            Spectrum::RgbUnbounded(s) => s.sample(lambda),
            Spectrum::RgbIlluminant(s) => s.sample(lambda),
        }
    }
}


#[derive(Debug, PartialEq, Clone)]
pub struct ConstantSpectrum {
    c: Float,
}

impl ConstantSpectrum {
    #[inline]
    pub const fn new(c: Float) -> ConstantSpectrum {
        ConstantSpectrum { c }
    }
}

impl AbstractSpectrum for ConstantSpectrum {
    fn get(&self, _lambda: Float) -> Float {
        self.c
    }

    fn max_value(&self) -> Float {
        self.c
    }

    fn sample(&self, _lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::from_const(self.c)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct DenselySampledSpectrum {
    lambda_min: i32,
    lambda_max: i32,
    values: Vec<Float>
}

impl DenselySampledSpectrum {
    pub fn new(spectrum: &Spectrum) -> DenselySampledSpectrum {
        Self::new_range(spectrum, LAMBDA_MIN as i32, LAMBDA_MAX as i32)
    }

    pub fn new_range(spectrum: &Spectrum, lambda_min: i32, lambda_max: i32) -> DenselySampledSpectrum {
        let values: Vec<Float> = (lambda_min..=lambda_max).map(|lambda: i32| spectrum.get(lambda as Float)).collect();
        DenselySampledSpectrum {
            lambda_min,
            lambda_max,
            values,
        }
    }

    pub fn sample_fn(f: impl Fn(Float) -> Float, lambda_min: usize, lambda_max: usize) -> DenselySampledSpectrum {
        let mut values = vec![0.0; lambda_max - lambda_min + 1];
        for lambda in lambda_min..=lambda_max {
            values[lambda - lambda_min] = f(lambda as Float);
        }
        DenselySampledSpectrum {
            values,
            lambda_min: lambda_min as i32,
            lambda_max: lambda_max as i32,
        }
    }

    pub fn d(temperature: Float) -> DenselySampledSpectrum {
        let cct = temperature * 1.4388 / 1.4380;

        if cct < 4000.0 {
            let bb = BlackbodySpectrum::new(cct);
            let blackbody = DenselySampledSpectrum::sample_fn(
                |lambda: Float| bb.get(lambda),
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            return blackbody;
        }

        let x = if cct <= 7000.0 {
            -4.607 * 1e9 / Float::powi(cct, 3)
                + 2.9678 * 1e6 / cct * cct
                + 0.09911 * 1e3 / cct
                + 0.244063
        } else {
            -2.0064 * 1e9 / Float::powi(cct, 3)
                + 1.9018 * 1e6 / cct * cct
                + 0.24748 * 1e3 / cct
                + 0.23704
        };

        let y = -3.0 * x * x + 2.870 * x - 0.275;

        let m = 0.0241 + 0.2562 * x - 0.7341 * y;
        let m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m;
        let m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m;

        let mut values: Vec<Float> = Vec::with_capacity(NUM_CIES_SAMPLES);
        for i in 0..NUM_CIES_SAMPLES {
            values.push((CIE_S0[i] + CIE_S1[i] * m1 + CIE_S2[i] * m2) * 0.01);
        }

        let dpls = &Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::new(
            CIE_S_LAMBDA.as_slice(),
            values.as_slice(),
        ));

        DenselySampledSpectrum::new(dpls)
    }

    pub fn scale(&mut self, s: Float) {
        for v in &mut self.values {
            *v *= s;
        }
    }
}

impl AbstractSpectrum for DenselySampledSpectrum {
    fn get(&self, lambda: Float) -> Float {
        let offset = lambda as i32 - self.lambda_min;
        if offset < 0 || offset >= self.values.len() as i32 {
            return 0.0;
        }
        self.values[offset as usize]
    }

    fn max_value(&self) -> Float {
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        if max.is_nan() {
            panic!("Empty DenselySampledSpectrum");
        }
        max
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            let offset: i32 = lambda[i].round() as i32 - self.lambda_min;
            s[i] = if offset < 0 || offset >= self.values.len() as i32 {
                0.0
            } else {
                self.values[offset as usize]
            }
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PiecewiseLinearSpectrum {
    lambdas: Vec<Float>,
    values: Vec<Float>,
}

impl PiecewiseLinearSpectrum {
    pub fn new(lambdas: &[Float], values: &[Float]) -> PiecewiseLinearSpectrum {
        debug_assert_eq!(lambdas.len(), values.len());

        let mut l = vec![0.0; lambdas.len()];
        l.copy_from_slice(lambdas);
        let mut v = vec![0.0; values.len()];
        v.copy_from_slice(values);

        debug_assert!(l.windows(2).all(|p| p[0] <= p[1]));

        PiecewiseLinearSpectrum {
            lambdas: l,
            values: v,
        }
    }

    pub fn from_interleaved<const N: usize, const S: usize>(
        samples: &[Float; N],
        normalize: bool,
    ) -> PiecewiseLinearSpectrum {
        debug_assert_eq!(N / 2, S);
        debug_assert_eq!(0, samples.len() % 2);

        let n = samples.len() / 2;
        let mut lambda = Vec::new();
        let mut v = Vec::new();

        if samples[0] > LAMBDA_MIN {
            lambda.push(LAMBDA_MIN - 1.0);
            v.push(samples[1]);
        }

        for i in 0..n {
            lambda.push(samples[2 * i]);
            v.push(samples[2 * i + 1]);

            if i > 0 {
                debug_assert!(*lambda.last().unwrap() > lambda[lambda.len() - 2]);
            }
        }

        if *lambda.last().unwrap() < LAMBDA_MAX {
            lambda.push(LAMBDA_MAX + 1.0);
            v.push(*v.last().unwrap());
        }

        #[allow(clippy::useless_conversion)]
        let mut spectrum = PiecewiseLinearSpectrum::new(
            lambda.as_slice().try_into().expect("invalid length"),
            v.as_slice().try_into().expect("invalid length")
        );

        if normalize {
            spectrum.scale(
                CIE_Y_INTEGRAL
                    / inner_product::<PiecewiseLinearSpectrum, Spectrum>(
                        &spectrum,
                        Spectrum::get_cie(Cie::Y),
                    ),
            );
        }

        spectrum
    }

    pub fn read(filename: &str) -> Option<PiecewiseLinearSpectrum> {
        let vals = read_float_file(filename);
        if vals.is_empty() {
            warn!(@image filename, "unable to read spectrum file");
            return None;
        }

        if vals.len() % 2 != 0 {
            warn!(@image filename, "extra value found in spectrum file");
            return None;
        }

        let mut lambdas = Vec::new();
        let mut values = Vec::new();
        
        for i in 0..(vals.len() / 2) {
            if i > 0 && vals[2 * i] <= *lambdas.last().unwrap() {
                warn!(
                    @image filename,
                    "spectrum file invalid at {} entry, wavelengths not increasing ({} >= {})",
                    i, *lambdas.last().unwrap(), vals[2 * i],
                );
                return None;
            }

            lambdas.push(vals[2 * i]);
            values.push(vals[2 * i + 1]);
        }

        Some(PiecewiseLinearSpectrum {
            lambdas,
            values,
        })
    }

    pub fn scale(&mut self, s: Float) {
        for v in &mut self.values {
            *v *= s;
        }
    }
}

impl AbstractSpectrum for PiecewiseLinearSpectrum {
    fn get(&self, lambda: Float) -> Float {
        if self.lambdas.is_empty()
        || lambda < *self.lambdas.first().unwrap()
        || lambda > *self.lambdas.last().unwrap() {
            return 0.0;
        }

        let o = find_interval(self.lambdas.len(), |i| -> bool {
            self.lambdas[i] <= lambda
        });

        debug_assert!(lambda >= self.lambdas[o] && lambda <= self.lambdas[o + 1]);

        let t = (lambda - self.lambdas[o]) / (self.lambdas[o + 1] - self.lambdas[o]);
        lerp(self.values[o], self.values[o + 1], t)
    }

    fn max_value(&self) -> Float {
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        if max.is_nan() {
            panic!("empty or NaN-filled spectrum");
        }

        max
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BlackbodySpectrum {
    t: Float,
    normalization_factor: Float,
}

impl BlackbodySpectrum {
    pub fn new(t: Float) -> BlackbodySpectrum {
        let lambda_max = 2.8977721e-3 / t;
        let normalization_factor = 1.0 / BlackbodySpectrum::blackbody(lambda_max * 1e9, t);
        BlackbodySpectrum { t, normalization_factor }
    }

    fn blackbody(lambda: Float, temperature: Float) -> Float {
        if temperature < 0.0 {
            return 0.0;
        }

        let c = 299792458.0;
        let h = 6.62606957e-34;
        let kb = 1.3806488e-23;
        let l = lambda * 1e-9;

        let le = (2.0 * h * c * c) / (l.powi(5) * (Float::exp((h * c) / (l * kb * temperature)) - 1.0));

        debug_assert!(!le.is_nan());

        le
    }
}

impl AbstractSpectrum for BlackbodySpectrum {
    fn get(&self, lambda: Float) -> Float {
        BlackbodySpectrum::blackbody(lambda, self.t) * self.normalization_factor
    }

    fn max_value(&self) -> Float {
        1.0
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = BlackbodySpectrum::blackbody(lambda[i], self.t) * self.normalization_factor;
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct RgbAlbedoSpectrum {
    rsp: RgbSigmoidPolynomial,
}

impl RgbAlbedoSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &Rgb) -> RgbAlbedoSpectrum {
        debug_assert!(Float::max(Float::max(rgb.r, rgb.g), rgb.b) <= 1.0);
        debug_assert!(Float::min(Float::min(rgb.r, rgb.g), rgb.b) >= 0.0);

        RgbAlbedoSpectrum {
            rsp: cs.to_rgb_coeffs(rgb),
        }
    }
}

impl AbstractSpectrum for RgbAlbedoSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.rsp.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.rsp.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct RgbUnboundedSpectrum {
    scale: Float,
    rsp: RgbSigmoidPolynomial,
}

impl RgbUnboundedSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &Rgb) -> RgbUnboundedSpectrum {
        let m = Float::max(Float::max(rgb.r, rgb.g), rgb.b);
        let scale = 2.0 * m;
        let rsp = if scale != 0.0 {
            cs.to_rgb_coeffs(&(rgb / scale))
        } else {
            cs.to_rgb_coeffs(&Rgb::new(0.0, 0.0, 0.0))
        };
        RgbUnboundedSpectrum { scale, rsp }
    }
}

impl AbstractSpectrum for RgbUnboundedSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.scale * self.rsp.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.scale * self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct RgbIlluminantSpectrum {
    scale: Float,
    rsp: RgbSigmoidPolynomial,
    illuminant: Arc<Spectrum>,
}

impl RgbIlluminantSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &Rgb) -> RgbIlluminantSpectrum {
        let m = Float::max(Float::max(rgb.r, rgb.g), rgb.b);
        let scale = 2.0 * m;
        let rsp = if scale != 0.0 {
            cs.to_rgb_coeffs(&(rgb / scale))
        } else {
            cs.to_rgb_coeffs(&Rgb::new(0.0, 0.0, 0.0))
        };
        RgbIlluminantSpectrum { scale, rsp, illuminant: cs.illuminant.clone() }
    }
}

impl AbstractSpectrum for RgbIlluminantSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.scale * self.rsp.get(lambda) * self.illuminant.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value() * self.illuminant.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.scale * self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s) * self.illuminant.sample(lambda)
    }
}


pub fn inner_product<T: AbstractSpectrum, U: AbstractSpectrum>(a: &T, b: &U) -> Float {
    let mut integral = 0.0;
    for lambda in (LAMBDA_MIN as i32)..=(LAMBDA_MAX as i32) {
        integral += a.get(lambda as Float) * b.get(lambda as Float);
    }
    integral
}


pub fn spectrum_to_photometric(mut s: &Spectrum) -> Float {
    if let Spectrum::RgbIlluminant(illum) = s {
        s = illum.illuminant.as_ref();
    }

    let mut y = 0.0;
    for lambda in (LAMBDA_MIN as i32)..=(LAMBDA_MAX as i32) {
        y += Spectrum::get_cie(Cie::Y).get(lambda as Float) * s.get(lambda as Float);
    }

    y
}

pub fn spectrum_to_xyz(mut s: &Spectrum) -> Xyz {
    Xyz::new(
        inner_product(Spectrum::get_cie(Cie::X), s),
        inner_product(Spectrum::get_cie(Cie::Y), s),
        inner_product(Spectrum::get_cie(Cie::Z), s),
    ) / CIE_Y_INTEGRAL
}
