use std::ops::{Index, IndexMut};

use super::{sampled::{SampledSpectrum, NUM_SPECTRUM_SAMPLES}, spectrum::{LAMBDA_MAX, LAMBDA_MIN}};
use crate::math::*;


#[derive(Debug)]
pub struct SampledWavelengths {
    lambda: [Float; NUM_SPECTRUM_SAMPLES],
    pdf: [Float; NUM_SPECTRUM_SAMPLES]
}

impl SampledWavelengths {
    pub fn sample_uniform(u: Float) -> SampledWavelengths {
        Self::sample_uniform_range(u, LAMBDA_MIN, LAMBDA_MAX)
    }

    pub fn sample_uniform_range(u: Float, lambda_min: Float, lambda_max: Float) -> SampledWavelengths {
        debug_assert!((0.0..=1.0).contains(&u));

        let mut lambda = [0.0; NUM_SPECTRUM_SAMPLES];
        lambda[0] = lerp(lambda_min, lambda_max, u);

        let delta = (lambda_max - lambda_min) / (NUM_SPECTRUM_SAMPLES as Float);
        for i in 1..NUM_SPECTRUM_SAMPLES {
            lambda[i] = lambda[i - 1] + delta;
            if lambda[i] > lambda_max {
                lambda[i] = lambda_min + (lambda[i] - lambda_max);
            }
        }

        let mut pdf = [0.0; NUM_SPECTRUM_SAMPLES];
        for v in pdf.iter_mut() {
            *v = 1.0 / (lambda_max - lambda_min);
        }

        SampledWavelengths { lambda, pdf }
    }

    pub fn sample_visible(u: Float) -> SampledWavelengths {
        let mut lambda = [0.0; NUM_SPECTRUM_SAMPLES];
        let mut pdf = [0.0; NUM_SPECTRUM_SAMPLES];

        for i in 0..NUM_SPECTRUM_SAMPLES {
            let mut up = u + (i as Float) / (NUM_SPECTRUM_SAMPLES as Float);
            if up > 1.0 {
                up -= 1.0;
            }
            lambda[i] = sample_visible_wavelengths(up);
            pdf[i] = visible_wavelengths_pdf(lambda[i]);
        }
        SampledWavelengths { lambda, pdf }
    }

    pub fn pdf(&self) -> SampledSpectrum {
        SampledSpectrum::new(self.pdf)
    }

    pub fn terminate_secondary(&mut self) {
        if self.secondary_terminated() {
            return;
        }
        for i in 1..NUM_SPECTRUM_SAMPLES {
            self.pdf[i] = 0.0;
        }
        self.pdf[0] /= NUM_SPECTRUM_SAMPLES as Float;
    }

    pub fn secondary_terminated(&self) -> bool {
        for i in 1..NUM_SPECTRUM_SAMPLES {
            if self.pdf[i] != 0.0 {
                return false;
            }
        }
        true
    }
}


impl Index<usize> for SampledWavelengths {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        self.lambda.index(index)
    }
}

impl IndexMut<usize> for SampledWavelengths {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.lambda.index_mut(index)
    }
}


pub fn sample_visible_wavelengths(u: Float) -> Float {
    538.0 - 138.888889 * Float::atanh(0.85691062 - 1.82750197 * u)
}

pub fn visible_wavelengths_pdf(lambda: Float) -> Float {
    if !(360.0..=830.0).contains(&lambda) {
        return 0.0;
    }
    let x = Float::cosh(0.0072 * (lambda - 538.0));
    0.0039398042 / (x * x)
}
