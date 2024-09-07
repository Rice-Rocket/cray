use std::ops::{Deref, Index, IndexMut};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use numeric::HasNan;

use crate::math::*;

use super::{cie::{Cie, CIE_Y_INTEGRAL}, colorspace::RgbColorSpace, rgb_xyz::{Rgb, Xyz}, spectrum::{Spectrum, AbstractSpectrum as _}, wavelengths::SampledWavelengths};


pub const NUM_SPECTRUM_SAMPLES: usize = 4;


#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct SampledSpectrum {
    pub values: [Float; NUM_SPECTRUM_SAMPLES],
}

impl SampledSpectrum {
    pub const fn new(values: [Float; NUM_SPECTRUM_SAMPLES]) -> SampledSpectrum {
        SampledSpectrum { values }
    }

    pub fn from_const(c: Float) -> SampledSpectrum {
        SampledSpectrum {
            values: [c; NUM_SPECTRUM_SAMPLES],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|x: &Float| *x == Float::ZERO)
    }

    pub fn safe_div(&self, other: &SampledSpectrum) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = if other[i] != 0.0 {
                self[i] / other[i]
            } else {
                0.0
            }
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn clamp(&self, min: Float, max: Float) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = self.values[i].clamp(min, max);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn clamp_zero(&self) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = Float::max(0.0, self.values[i])
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn powf(&self, e: Float) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = self.values[i].powf(e);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn powi(&self, e: i32) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = self.values[i].powi(e);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn exp(&self) -> SampledSpectrum {
        // TODO consider a similar elementwise FastExp().
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = self.values[i].exp()
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn sqrt(&self) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for (i, res) in result.iter_mut().enumerate() {
            *res = self.values[i].sqrt();
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn lerp(&self, other: &SampledSpectrum, t: Float) -> SampledSpectrum {
        (1.0 - t) * self + t * other
    }

    pub fn average(&self) -> Float {
        self.values.iter().sum::<Float>() / (self.values.len() as Float)
    }

    pub fn min_component_value(&self) -> Float {
        debug_assert!(!self.values.has_nan());
        let min = self.values.iter().fold(Float::NAN, |a, &b| a.min(b));
        debug_assert!(!min.is_nan());
        min
    }

    pub fn max_component_value(&self) -> Float {
        debug_assert!(!self.values.has_nan());
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        debug_assert!(!max.is_nan());
        max
    }

    pub fn to_xyz(&self, lambda: &SampledWavelengths) -> Xyz {
        // Sample the X, Y, and Z matching curves at lambda
        let x = Spectrum::get_cie(Cie::X).sample(lambda);
        let y = Spectrum::get_cie(Cie::Y).sample(lambda);
        let z = Spectrum::get_cie(Cie::Z).sample(lambda);

        // Evaluate estimator to compute (x, y, z) coefficients.
        let pdf = lambda.pdf();
        Xyz::new(
            (x * self).safe_div(&pdf).average(),
            (y * self).safe_div(&pdf).average(),
            (z * self).safe_div(&pdf).average(),
        ) / CIE_Y_INTEGRAL
    }

    /// Similar to to_xyz(), but only computes the y value for when only
    /// luminance is needed, thereby saving computations.
    pub fn y(&self, lambda: &SampledWavelengths) -> Float {
        let ys = Spectrum::get_cie(Cie::Y).sample(lambda);
        let pdf = lambda.pdf();
        (ys * self).safe_div(&pdf).average() / CIE_Y_INTEGRAL
    }

    pub fn to_rgb(&self, lambda: &SampledWavelengths, cs: &RgbColorSpace) -> Rgb {
        let xyz = self.to_xyz(lambda);
        cs.to_rgb(&xyz)
    }
}

impl Index<usize> for SampledSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        self.values.index(index)
    }
}

impl IndexMut<usize> for SampledSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.values.index_mut(index)
    }
}

// We implement Deref so that we can use the array's iter().
impl Deref for SampledSpectrum {
    type Target = [Float; NUM_SPECTRUM_SAMPLES];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl_op_ex!(+|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum
{
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s1[i] + s2[i];
    }
    debug_assert!(!result.contains(&Float::NAN));
    SampledSpectrum::new(result)
});

impl_op_ex!(
    -|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] - s2[i];
        }
        debug_assert!(!result.contains(&Float::NAN));
        SampledSpectrum::new(result)
    }
);

impl_op_ex!(
    *|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] * s2[i];
        }
        debug_assert!(!result.contains(&Float::NAN));
        SampledSpectrum::new(result)
    }
);

impl_op_ex_commutative!(*|s1: &SampledSpectrum, v: &Float| -> SampledSpectrum {
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES {
        result[i] = s1[i] * v;
    }
    debug_assert!(!result.contains(&Float::NAN));
    SampledSpectrum::new(result)
});

impl_op_ex!(/|s: &SampledSpectrum, v: &Float| -> SampledSpectrum
{
    debug_assert!(*v != 0.0);
    debug_assert!(!v.is_nan());
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s[i] / v;
    }
    debug_assert!(!result.contains(&Float::NAN));
    SampledSpectrum::new(result)
});

impl_op_ex!(/ |s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum
{
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s1[i] / s2[i];
    }
    debug_assert!(!result.contains(&Float::NAN));
    SampledSpectrum::new(result)
});

impl_op_ex!(+= |s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] += s2[i];
    }
});

impl_op_ex!(-= |s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] -= s2[i];
    }
});

impl_op_ex!(*= |s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] *= s2[i];
    }
});

impl_op_ex!(/= |s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] /= s2[i];
    }
});

impl_op_ex!(/= |s1: &mut SampledSpectrum, v: &Float|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] /= v;
    }
});

impl HasNan for SampledSpectrum {
    const NAN: Self = SampledSpectrum::new([Float::NAN; NUM_SPECTRUM_SAMPLES]);

    fn has_nan(&self) -> bool {
        self.values.iter().any(|x| x.is_nan())
    }

    fn has_finite(&self) -> bool {
        self.values.iter().all(|x| x.is_finite())
    }
}
