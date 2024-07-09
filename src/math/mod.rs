#![allow(dead_code)]

pub mod interaction;
pub mod bounds;
pub mod dim;
pub mod numeric;
pub mod ray;
pub mod sphere;
pub mod transform;
pub mod mat;
pub mod vect;
pub mod swizzle;
pub mod interval;
pub mod frame;
pub mod sampling;
pub mod tile;
pub mod vec2d;

pub use mat::{TMat2, TMat3, TMat4};
pub use vect::{TVec2, TVec3, TVec4, TPoint2, TPoint3, TPoint4, TNormal2, TNormal3};
pub use bounds::{direction::DirectionCone, Bounds2f, Bounds3f, Bounds2i, Bounds3i, Bounds2u, Bounds3u, TBounds2, TBounds3};

pub use frame::Frame;
pub use sphere::OctahedralVec3;
pub use dim::Dimension;
pub use interval::{Interval, FloatInterval};

pub use numeric::{Numeric, NumericNegative, NumericFloat, NumericField};
pub use ray::{Ray, RayDifferential, AuxiliaryRays};


// Pbrt B.2 Mathematical Infrastructure

use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

pub use super::sphere::*;

/// π
pub const PI: Scalar = 3.14159265358979323846;
/// 2π
pub const TAU: Scalar = 6.28318530717958647693;
/// 1/π
pub const FRAC_1_PI: Scalar = 0.31830988618379067154;
/// 1/2π
pub const FRAC_1_TAU: Scalar = 0.15915494309189533577;
/// 1/4π
pub const FRAC_1_4PI: Scalar = 0.07957747154594766788;
/// π/2
pub const FRAC_PI_2: Scalar = 1.57079632679489661923;
/// π/4
pub const FRAC_PI_4: Scalar = 0.78539816339744830961;
/// √2
pub const SQRT_2: Scalar = 1.41421356237309504880;

/// Computse the linear interpolation between `a` and `b` at input `t`.
pub fn lerp<T: std::ops::Add<T, Output = T> + std::ops::Sub<T, Output = T> + std::ops::Mul<T, Output = T> + Copy>(
    a: T,
    b: T,
    t: T,
) -> T {
    a + (b - a) * t
}

pub fn lerp_float<T>(a: T, b: T, t: Scalar) -> T
where
    T: Add<T, Output = T>,
    T: Mul<Scalar, Output = T>,
{
    a * (1.0 - t) + b * t
}

/// Computes the smoothstep function at a given input `t` with bounds `a`
/// and `b`.
#[inline]
pub fn smoothstep(a: Scalar, b: Scalar, t: Scalar) -> Scalar {
    if a == b {
        return if t < a { 0.0 } else { 1.0 };
    };
    let k = ((t - a) / (b - a)).clamp(0.0, 1.0);
    k * k * (3.0 - 2.0 * k)
}

/// Converts the given degrees `deg` to radians.
#[inline]
pub fn to_radians(deg: Scalar) -> Scalar {
    (PI / 180.0) * deg
}
/// Converts the given radians `rad` to degrees.
#[inline]
pub fn to_degrees(rad: Scalar) -> Scalar {
    (180.0 / PI) * rad
}

/// Computes the square of a number.
#[inline]
pub fn sqr<T: Numeric + Clone + Copy + Mul<T, Output = T>>(x: T) -> T {
    x * x
}

/// Computes the `sinc(x)` (`sin(x)/x`) function defined at 0.
#[inline]
pub fn sinc(x: Scalar) -> Scalar {
    if 1.0 - x * x == 1.0 {
        return 1.0;
    }
    x.sin() / x
}

/// Get the bits of a floating point number.
#[inline]
pub fn float_to_bits(f: Scalar) -> ScalarAsBits {
    let rui: ScalarAsBits;
    unsafe {
        let ui: ScalarAsBits = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Convert the bit representation of a float back into the value.
#[inline]
pub fn bits_to_float(ui: ScalarAsBits) -> Scalar {
    let rf: Scalar;
    unsafe {
        let f: Scalar = std::mem::transmute_copy(&ui);
        rf = f;
    }
    rf
}

pub fn next_float_up(mut v: Scalar) -> Scalar {
    if v.is_infinite() && v > 0.0 {
        return v;
    }
    if v == -0.0 {
        v = 0.0;
    }

    let mut ui: ScalarAsBits = float_to_bits(v);
    if v >= 0.0 {
        ui += 1;
    } else {
        ui -= 1;
    }
    bits_to_float(ui)
}

pub fn next_float_down(mut v: Scalar) -> Scalar {
    if v.is_infinite() && v < 0.0 {
        return v;
    }
    if v == 0.0 {
        v = -0.0;
    }
    let mut ui: ScalarAsBits = float_to_bits(v);
    if v > 0.0 {
        ui -= 1;
    } else {
        ui += 1;
    }
    bits_to_float(ui)
}

/// Returns the exponent of the bit representation of the float. (+127)
#[inline]
pub fn exponent(v: Scalar) -> ScalarAsBits {
    float_to_bits(v) >> 23
}

/// Returns the mantissa of the bit representation of the float.
#[inline]
pub fn significand(v: Scalar) -> ScalarAsBits {
    float_to_bits(v) & ((1 << 23) - 1)
}

/// Computes the integer component of `log2(x)`.
#[inline]
pub fn log2int(v: Scalar) -> i32 {
    if v < 1.0 {
        return -log2int(1.0 / v);
    };
    exponent(v) as i32 + (if significand(v) >= 0b00000000001101010000010011110011 { 1 } else { 0 })
}

/// Computes the integer component of `log4(x)`.
#[inline]
pub fn log4int(v: Scalar) -> i32 {
    log2int(v) / 2
}

/// Computes the Gauss error function
#[inline]
pub fn erf(mut x: Scalar) -> Scalar {
    const A1: Scalar = 0.254829592;
    const A2: Scalar = -0.284496736;
    const A3: Scalar = 1.421413741;
    const A4: Scalar = -1.453152027;
    const A5: Scalar = 1.061405429;
    const P: Scalar = 0.3275911;

    let sign = x.signum();
    x = x.abs();

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Computes the guassian distribution at the input `x`.
///
/// Defaults for `mu` and `sigma` should be 0 and 1 respectively.
#[inline]
pub fn gaussian(x: Scalar, mu: Scalar, sigma: Scalar) -> Scalar {
    1.0 / (2.0 * PI * sigma * sigma).sqrt() * (-sqr(x - mu) / (2.0 * sigma * sigma).exp())
}

/// Computes the integral of the gaussian distribution at the input `x`.
///
/// Defaults for `mu` and `sigma` should be 0 and 1 respectively.
#[inline]
pub fn gaussian_integral(x0: Scalar, x1: Scalar, mu: Scalar, sigma: Scalar) -> Scalar {
    let sigma_root_2 = sigma * SQRT_2;
    0.5 * (erf((mu - x0) / sigma_root_2) - erf((mu - x1) / sigma_root_2))
}

/// Computes the logistic distribution at the input `x` and a scale factor
/// `s`.
#[inline]
pub fn logistic(mut x: Scalar, s: Scalar) -> Scalar {
    x = x.abs();
    (-x / s).exp() / (s * sqr(1.0 + (-x / s).exp()))
}

/// Computes the cumulative distribution function (CDF) of the logistic
/// distribution at the input `x` and a scale factor `s`.
#[inline]
pub fn logistic_cdf(x: Scalar, s: Scalar) -> Scalar {
    1.0 / (1.0 + (-x / s).exp())
}

/// The logistic function limited to the interval `[a, b]` and renormalized.
#[inline]
pub fn trimmed_logistic(x: Scalar, s: Scalar, a: Scalar, b: Scalar) -> Scalar {
    logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
}

#[inline]
pub fn sin_cos(x: Scalar) -> (Scalar, Scalar) {
    Scalar::sin_cos(x)
}

#[inline]
pub fn find_interval(size: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut first = 1;
    let mut last = size as i32 - 2;
    while last > 0 {
        let half = last >> 1;
        let middle = first + half;
        let pred_result = pred(middle.try_into().unwrap());
        first = if pred_result { middle + 1 } else { first };
        last = if pred_result { last - (half + 1) } else { half };
    }

    i32::clamp(first - 1, 0, size as i32 - 2) as usize
}

pub fn linear_least_squares_3<const ROWS: usize>(
    a: &[[Scalar; 3]; ROWS],
    b: &[[Scalar; 3]; ROWS],
) -> Option<Mat3> {
    let (at_a, at_b) = linear_least_squares_helper::<3, ROWS, Mat3>(a, b);
    let at_ai = at_a.try_inverse()?;
    Some((at_ai * at_b).transpose())
}

pub fn linear_least_squares_helper<const N: usize, const ROWS: usize, M>(
    a: &[[Scalar; N]; ROWS],
    b: &[[Scalar; N]; ROWS],
) -> (M, M)
where 
    M: Default + IndexMut<(usize, usize)> + Index<(usize, usize), Output = Scalar>
{
    let mut a_t_a = M::default();
    let mut a_t_b = M::default();

    for i in 0..N {
        for j in 0..N {
            for r in 0..ROWS {
                a_t_a[(i, j)] += a[r][i] * a[r][j];
                a_t_b[(i, j)] += a[r][i] * b[r][j];
            }
        }
    }

    (a_t_a, a_t_b)
}

pub fn modulo<T>(a: T, b: T) -> T
where
    T: Div<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy + std::cmp::PartialOrd<i32>,
{
    let result = a - (a / b) * b;
    if result < 0
    {
        result + b
    }
    else
    {
        result
    }
}

pub fn windowed_sinc(x: Scalar, radius: Scalar, tau: Scalar) -> Scalar {
    if x.abs() > radius {
        return 0.0;
    }

    sinc(x) * sinc(x / tau)
}


pub mod safe {
    use crate::{math::{Numeric, Scalar}, NumericFloat};

    #[inline]
    pub fn sqrt<T: Numeric + NumericFloat>(x: T) -> T {
        (x.nmax(T::ZERO)).nsqrt()
    }

    #[inline]
    pub fn asin(x: Scalar) -> Scalar {
        x.clamp(-1.0, 1.0).asin()
    }

    #[inline]
    pub fn acos(x: Scalar) -> Scalar {
        x.clamp(-1.0, 1.0).acos()
    }
}


pub type Scalar = f32;
pub type ScalarAsBits = u32;

pub type Mat2 = TMat2<Scalar>;
pub type Mat3 = TMat3<Scalar>;
pub type Mat4 = TMat4<Scalar>;

pub type Mat2i = TMat2<Interval>;
pub type Mat3i = TMat3<Interval>;
pub type Mat4i = TMat4<Interval>;

pub type Point2f = TPoint2<Scalar>;
pub type Point3f = TPoint3<Scalar>;
pub type Point4f = TPoint4<Scalar>;
pub type Point2fi = TPoint2<Interval>;
pub type Point3fi = TPoint3<Interval>;
pub type Point4fi = TPoint4<Interval>;

pub type Point2i = TPoint2<i32>;
pub type Point3i = TPoint3<i32>;
pub type Point4i = TPoint4<i32>;
pub type Point2u = TPoint2<u32>;
pub type Point3u = TPoint3<u32>;
pub type Point4u = TPoint4<u32>;

pub type Vec2f = TVec2<Scalar>;
pub type Vec3f = TVec3<Scalar>;
pub type Vec4f = TVec4<Scalar>;
pub type Vec2fi = TVec2<Interval>;
pub type Vec3fi = TVec3<Interval>;
pub type Vec4fi = TVec4<Interval>;

pub type Vec2i = TVec2<i32>;
pub type Vec3i = TVec3<i32>;
pub type Vec4i = TVec4<i32>;
pub type Vec2u = TVec2<u32>;
pub type Vec3u = TVec3<u32>;
pub type Vec4u = TVec4<u32>;

pub type Normal2f = TNormal2<Scalar>;
pub type Normal3f = TNormal3<Scalar>;
pub type Normal2fi = TNormal2<Interval>;
pub type Normal3fi = TNormal3<Interval>;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        assert!((erf(0.0)).abs() < f32::EPSILON);
        assert!((erf(-0.5) + 0.5205).abs() < f32::EPSILON);
        assert!((erf(0.5) - 0.5205).abs() < f32::EPSILON);
    }
}
