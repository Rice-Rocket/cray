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
pub mod scattering;

use fast_polynomial::poly;
pub use mat::{TMat2, TMat3, TMat4};
pub use vect::{Vec2, Vec3, Vec4, Point2, Point3, Point4, Normal2, Normal3, Dot};
pub use bounds::{direction::DirectionCone, Bounds2f, Bounds3f, Bounds2i, Bounds3i, Bounds2u, Bounds3u, Bounds2, Bounds3};

pub use frame::Frame;
pub use sphere::OctahedralVec3;
pub use dim::Dimension;
pub use interval::{Interval, FloatInterval};

pub use numeric::{NumericConsts, NumericNegative, NumericFloat, NumericField, NumericOrd, DifferenceOfProducts};
pub use ray::{Ray, RayDifferential, AuxiliaryRays};


// Pbrt B.2 Mathematical Infrastructure

use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

pub use super::sphere::*;

/// π
pub const PI: Float = 3.14159265358979323846;
/// 2π
pub const TAU: Float = 6.28318530717958647693;
/// 1/π
pub const FRAC_1_PI: Float = 0.31830988618379067154;
/// 1/2π
pub const FRAC_1_TAU: Float = 0.15915494309189533577;
/// 1/4π
pub const FRAC_1_4PI: Float = 0.07957747154594766788;
/// π/2
pub const FRAC_PI_2: Float = 1.57079632679489661923;
/// π/4
pub const FRAC_PI_4: Float = 0.78539816339744830961;
/// √2
pub const SQRT_2: Float = 1.41421356237309504880;
pub const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;

/// Computse the linear interpolation between `a` and `b` at input `t`.
pub fn lerp<T: std::ops::Add<T, Output = T> + std::ops::Sub<T, Output = T> + std::ops::Mul<T, Output = T> + Copy>(
    a: T,
    b: T,
    t: T,
) -> T {
    a + (b - a) * t
}

pub fn lerp_float<T>(a: T, b: T, t: Float) -> T
where
    T: Add<T, Output = T>,
    T: Mul<Float, Output = T>,
{
    a * (1.0 - t) + b * t
}

/// Computes the smoothstep function at a given input `t` with bounds `a`
/// and `b`.
#[inline]
pub fn smoothstep(a: Float, b: Float, t: Float) -> Float {
    if a == b {
        return if t < a { 0.0 } else { 1.0 };
    };
    let k = ((t - a) / (b - a)).clamp(0.0, 1.0);
    k * k * (3.0 - 2.0 * k)
}

/// Converts the given degrees `deg` to radians.
#[inline]
pub fn to_radians(deg: Float) -> Float {
    (PI / 180.0) * deg
}
/// Converts the given radians `rad` to degrees.
#[inline]
pub fn to_degrees(rad: Float) -> Float {
    (180.0 / PI) * rad
}

/// Computes the square of a number.
#[inline]
pub fn sqr<T: Copy + Mul<T, Output = T>>(x: T) -> T {
    x * x
}

/// Computes the `sinc(x)` (`sin(x)/x`) function defined at 0.
#[inline]
pub fn sinc(x: Float) -> Float {
    if 1.0 - x * x == 1.0 {
        return 1.0;
    }
    x.sin() / x
}

/// Get the bits of a floating point number.
#[inline]
pub fn float_to_bits(f: Float) -> FloatAsBits {
    let rui: FloatAsBits;
    unsafe {
        let ui: FloatAsBits = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Convert the bit representation of a float back into the value.
#[inline]
pub fn bits_to_float(ui: FloatAsBits) -> Float {
    let rf: Float;
    unsafe {
        let f: Float = std::mem::transmute_copy(&ui);
        rf = f;
    }
    rf
}

#[inline]
pub fn mix_bits(mut v: u64) -> u64 {
    v ^= v.wrapping_shr(31);
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v.wrapping_shr(27);
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v.wrapping_shr(33);
    v
}

pub fn next_float_up(mut v: Float) -> Float {
    if v.is_infinite() && v > 0.0 {
        return v;
    }
    if v == -0.0 {
        v = 0.0;
    }

    let mut ui: FloatAsBits = float_to_bits(v);
    if v >= 0.0 {
        ui += 1;
    } else {
        ui -= 1;
    }
    bits_to_float(ui)
}

pub fn next_float_down(mut v: Float) -> Float {
    if v.is_infinite() && v < 0.0 {
        return v;
    }
    if v == 0.0 {
        v = -0.0;
    }
    let mut ui: FloatAsBits = float_to_bits(v);
    if v > 0.0 {
        ui -= 1;
    } else {
        ui += 1;
    }
    bits_to_float(ui)
}

/// Returns the exponent of the bit representation of the float. (+127)
#[inline]
pub fn exponent(v: Float) -> FloatAsBits {
    float_to_bits(v) >> 23
}

/// Returns the mantissa of the bit representation of the float.
#[inline]
pub fn significand(v: Float) -> FloatAsBits {
    float_to_bits(v) & ((1 << 23) - 1)
}

/// Computes the integer component of `log2(x)`.
#[inline]
pub fn log2int(v: Float) -> i32 {
    if v < 1.0 {
        return -log2int(1.0 / v);
    };
    exponent(v) as i32 + (if significand(v) >= 0b00000000001101010000010011110011 { 1 } else { 0 })
}

/// Computes the integer component of `log4(x)`.
#[inline]
pub fn log4int(v: Float) -> i32 {
    log2int(v) / 2
}

/// Computes the Gauss error function
#[inline]
pub fn erf(mut x: Float) -> Float {
    const A1: Float = 0.254829592;
    const A2: Float = -0.284496736;
    const A3: Float = 1.421413741;
    const A4: Float = -1.453152027;
    const A5: Float = 1.061405429;
    const P: Float = 0.3275911;

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
pub fn gaussian(x: Float, mu: Float, sigma: Float) -> Float {
    1.0 / (2.0 * PI * sigma * sigma).sqrt() * (-sqr(x - mu) / (2.0 * sigma * sigma).exp())
}

/// Computes the integral of the gaussian distribution at the input `x`.
///
/// Defaults for `mu` and `sigma` should be 0 and 1 respectively.
#[inline]
pub fn gaussian_integral(x0: Float, x1: Float, mu: Float, sigma: Float) -> Float {
    let sigma_root_2 = sigma * SQRT_2;
    0.5 * (erf((mu - x0) / sigma_root_2) - erf((mu - x1) / sigma_root_2))
}

/// Computes the logistic distribution at the input `x` and a scale factor
/// `s`.
#[inline]
pub fn logistic(mut x: Float, s: Float) -> Float {
    x = x.abs();
    (-x / s).exp() / (s * sqr(1.0 + (-x / s).exp()))
}

/// Computes the cumulative distribution function (CDF) of the logistic
/// distribution at the input `x` and a scale factor `s`.
#[inline]
pub fn logistic_cdf(x: Float, s: Float) -> Float {
    1.0 / (1.0 + (-x / s).exp())
}

/// The logistic function limited to the interval `[a, b]` and renormalized.
#[inline]
pub fn trimmed_logistic(x: Float, s: Float, a: Float, b: Float) -> Float {
    logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
}

pub fn quadratic(a: Float, b: Float, c: Float) -> Option<(Float, Float)> {
    if a == 0.0 {
        if b == 0.0 {
            return None;
        }

        return Some((-c / b, -c / b));
    }

    let discrim = Float::difference_of_products(b, b, 4.0 * a, c);
    if discrim < 0.0 {
        return None;
    }

    let root_discrim = Float::sqrt(discrim);

    let q = -0.5 * (b + Float::copysign(root_discrim, b));
    let t0 = q / a;
    let t1 = c / q;
    if t0 > t1 {
        Some((t1, t0))
    } else {
        Some((t0, t1))
    }
}

#[inline]
pub fn sin_cos(x: Float) -> (Float, Float) {
    Float::sin_cos(x)
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
    a: &[[Float; 3]; ROWS],
    b: &[[Float; 3]; ROWS],
) -> Option<Mat3> {
    let (at_a, at_b) = linear_least_squares_helper::<3, ROWS, Mat3>(a, b);
    let at_ai = at_a.try_inverse()?;
    Some((at_ai * at_b).transpose())
}

pub fn linear_least_squares_helper<const N: usize, const ROWS: usize, M>(
    a: &[[Float; N]; ROWS],
    b: &[[Float; N]; ROWS],
) -> (M, M)
where 
    M: Default + IndexMut<(usize, usize)> + Index<(usize, usize), Output = Float>
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

pub fn windowed_sinc(x: Float, radius: Float, tau: Float) -> Float {
    if x.abs() > radius {
        return 0.0;
    }

    sinc(x) * sinc(x / tau)
}

pub fn gamma(n: i32) -> Float {
    (n as Float * MACHINE_EPSILON) / (1.0 - n as Float * MACHINE_EPSILON)
}

pub fn difference_of_products_float_vec(a: Float, b: Vec3f, c: Float, d: Vec3f) -> Vec3f {
    let cd = c * d;
    let difference = a * b - cd;
    let error = -c * d + cd;
    difference + error
}

/// Default x_eps and f_eps = 1e-6
pub fn newton_bisection<T, F: Fn(Float, &mut T, &mut T) -> (Float, Float)>(
    mut x0: Float,
    mut x1: Float,
    f: F,
    x_eps: Float,
    f_eps: Float,
    fhat0: &mut T,
    fhat1: &mut T,
) -> Float {
    let fx0 = f(x0, fhat0, fhat1).0;
    let fx1 = f(x1, fhat0, fhat1).0;

    if fx0.abs() < f_eps { return x0 };
    if fx1.abs() < f_eps { return x1 };

    let mut start_is_negative = fx0 < 0.0;
    let mut x_mid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    loop {
        if !(x0 < x_mid && x_mid < x1) {
            x_mid = (x0 + x1) / 2.0;
        }

        let fxmid = f(x_mid, fhat0, fhat1);
        debug_assert!(!fxmid.0.is_nan());

        if start_is_negative == (fxmid.0 < 0.0) {
            x0 = x_mid;
        } else {
            x1 = x_mid;
        }

        if (x1 - x0) < x_eps || fxmid.0.abs() < f_eps {
            return x_mid;
        }

        x_mid -= fxmid.0 / fxmid.1;
    }
}

pub fn catmull_rom_weights(nodes: &[Float], x: Float, offset: &mut i32, weights: &mut [Float]) -> bool {
    debug_assert!(weights.len() >= 4);

    if nodes.is_empty() {
        return false;
    }

    if !(x >= *nodes.first().unwrap() && x <= *nodes.last().unwrap()) {
        return false;
    }

    let idx = find_interval(nodes.len(), |i| nodes[i] <= x);
    *offset = (idx - 1) as i32;
    let x0 = nodes[idx];
    let x1 = nodes[idx + 1];

    let t = (x - x0) / (x1 - x0);
    let t2 = t * t;
    let t3 = t2 * t;

    weights[1] = 2.0 * t3 - 3.0 * t2 + 1.0;
    weights[2] = -2.0 * t3 + 3.0 * t2;

    if idx > 0 {
        let w0 = (t3 - 2.0 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        let w0 = t3 - 2.0 * t2 + t;
        weights[0] = 0.0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    if idx + 2 < nodes.len() {
        let w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        let w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0;
    }

    true
}

pub fn sample_catmull_rom_2d(
    nodes1: &[Float],
    nodes2: &[Float],
    values: &[Float],
    cdf: &[Float],
    alpha: Float,
    mut u: Float,
    f_val: Option<&mut Float>,
    pdf: Option<&mut Float>,
) -> Float {
    let mut offset = 0;
    let mut weights = [0.0; 4];
    if !catmull_rom_weights(nodes1, alpha, &mut offset, &mut weights) {
        return 0.0;
    }

    let interpolate = |arr: &[Float], idx: usize| {
        let mut v = 0.0;
        for i in 0..4 {
            if weights[i] != 0.0 {
                v += arr[(offset as usize + i) * nodes2.len() + idx] * weights[i];
            }
        }
        v
    };

    let maximum = interpolate(cdf, nodes2.len() - 1);
    u *= maximum;
    let idx = find_interval(nodes2.len(), |i: usize| interpolate(cdf, i) <= u);

    let f0 = interpolate(values, idx);
    let f1 = interpolate(values, idx + 1);
    let x0 = nodes2[idx];
    let x1 = nodes2[idx + 1];
    let width = x1 - x0;

    u = (u - interpolate(cdf, idx)) / width;

    let d0 = if idx > 0 {
        width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1])
    } else {
        f1 - f0
    };

    let d1 = if idx + 2 < nodes2.len() {
        width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0)
    } else {
        f1 - f0
    };

    let mut fhat0 = 0.0;
    let mut fhat1 = 0.0;
    let t = newton_bisection(
        0.0,
        1.0,
        |t: Float, fhat0: &mut Float, fhat1: &mut Float| -> (Float, Float) {
            *fhat0 = poly(t, &[0.0, f0, 0.5 * d0, (1.0 / 3.0) * (-2.0 * d0 - d1) + f1 - f0, 0.25 * (d0 + d1) + 0.5 * (f0 - f1)]);
            *fhat1 = poly(t, &[f0, d0, -2.0 * d0 - d1 + 3.0 * (f1 - f0), d0 + d1 + 2.0 * (f0 - f1)]);
            (*fhat0 - u, *fhat1)
        },
        1e-6,
        1e-6,
        &mut fhat0,
        &mut fhat1,
    );

    if let Some(v) = f_val {
        *v = fhat1;
    }

    if let Some(pdf) = pdf {
        *pdf = fhat1 / maximum;
    }

    x0 + width * t
}

pub fn integrate_catmull_rom(nodes: &[Float], f: &[Float], cdf: &mut [Float]) -> Float {
    debug_assert!(nodes.len() == f.len());
    let mut sum = 0.0;
    cdf[0] = 0.0;

    for i in 0..nodes.len() - 1 {
        let x0 = nodes[i];
        let x1 = nodes[i + 1];
        let f0 = f[i];
        let f1 = f[i + 1];
        let width = x1 - x0;

        let d0 = if i > 0 { width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) } else { f1 - f0 };
        let d1 = if i + 2 < nodes.len() { width * (f[i + 2] - f0) / (nodes[i + 2] - x0) } else { f1 - f0 };

        sum += width * ((f0 + f1) / 2.0 + (d0 - d1) / 12.0);
        cdf[i + 1] = sum;
    }

    sum
}

pub fn invert_catmull_rom(nodes: &[Float], f: &[Float], u: Float) -> Float {
    if u <= *f.first().unwrap() {
        return *nodes.first().unwrap();
    } else if u >= *f.last().unwrap() {
        return *nodes.last().unwrap();
    }

    let i = find_interval(f.len(), |i| f[i] <= u);

    let x0 = nodes[i];
    let x1 = nodes[i + 1];
    let f0 = f[i];
    let f1 = f[i + 1];
    let width = x1 - x0;

    let d0 = if i > 0 { width * (f1 - f[i - 1]) / (x1 - nodes[i - 1]) } else { f1 - f0 };
    let d1 = if i + 2 < nodes.len() { width * (f[i + 2] - f0) / nodes[i + 2] - x0 } else { f1 - f0 };

    let t = newton_bisection(
        0.0,
        1.0,
        |t, _, _| {
            let t2 = t * t;
            let t3 = t2 * t;

            let fhat0 = (2.0 * t3 - 3.0 * t2 + 1.0) * f0 + (-2.0 * t3 + 3.0 * t2) * f1
                + (t3 - 2.0 * t2 + t) * d0 + (t3 - t2) * d1;

            let fhat1 = (6.0 * t2 - 6.0 * t) * f0 + (-6.0 * t2 + 6.0 * t) * f1
                + (3.0 * t2 - 4.0 * t + 1.0) * d0 + (3.0 * t2 - 2.0 * t) * d1;

            (fhat0 - u, fhat1)
        },
        1e-6,
        1e-6,
        &mut (),
        &mut (),
    );

    x0 + t * width
}

pub mod safe {
    use crate::{math::{NumericConsts, Float}, NumericFloat, NumericOrd};

    #[inline]
    pub fn sqrt<T: NumericConsts + NumericFloat + NumericOrd>(x: T) -> T {
        (x.nmax(T::ZERO)).nsqrt()
    }

    #[inline]
    pub fn asin(x: Float) -> Float {
        x.clamp(-1.0, 1.0).asin()
    }

    #[inline]
    pub fn acos(x: Float) -> Float {
        x.clamp(-1.0, 1.0).acos()
    }
}

#[cfg(not(feature = "use_f64"))]
pub type Float = f32;
#[cfg(not(feature = "use_f64"))]
pub type FloatAsBits = u32;

#[cfg(feature = "use_f64")]
pub type Float = f64;
#[cfg(feature = "use_f64")]
pub type FloatAsBits = u64;

pub type Mat2 = TMat2<Float>;
pub type Mat3 = TMat3<Float>;
pub type Mat4 = TMat4<Float>;

pub type Mat2i = TMat2<Interval>;
pub type Mat3i = TMat3<Interval>;
pub type Mat4i = TMat4<Interval>;

pub type Point2f = Point2<Float>;
pub type Point3f = Point3<Float>;
pub type Point4f = Point4<Float>;
pub type Point2fi = Point2<Interval>;
pub type Point3fi = Point3<Interval>;
pub type Point4fi = Point4<Interval>;

pub type Point2i = Point2<i32>;
pub type Point3i = Point3<i32>;
pub type Point4i = Point4<i32>;
pub type Point2u = Point2<u32>;
pub type Point3u = Point3<u32>;
pub type Point4u = Point4<u32>;

pub type Vec2f = Vec2<Float>;
pub type Vec3f = Vec3<Float>;
pub type Vec4f = Vec4<Float>;
pub type Vec2fi = Vec2<Interval>;
pub type Vec3fi = Vec3<Interval>;
pub type Vec4fi = Vec4<Interval>;

pub type Vec2i = Vec2<i32>;
pub type Vec3i = Vec3<i32>;
pub type Vec4i = Vec4<i32>;
pub type Vec2u = Vec2<u32>;
pub type Vec3u = Vec3<u32>;
pub type Vec4u = Vec4<u32>;

pub type Normal2f = Normal2<Float>;
pub type Normal3f = Normal3<Float>;
pub type Normal2fi = Normal2<Interval>;
pub type Normal3fi = Normal3<Interval>;


#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn test_erf() {
        assert!((erf(0.0)).abs() < Float::EPSILON);
        assert!((erf(-0.5) + 0.5205).abs() < Float::EPSILON);
        assert!((erf(0.5) - 0.5205).abs() < Float::EPSILON);
    }

    #[test]
    fn test_newton_bisection() {
        assert_approx_eq!(Float, 1.0, newton_bisection(
            0.0,
            10.0,
            |x, _, _| -> (Float, Float) {
                (-1.0 + x, 1.0)
            },
            1e-6,
            1e-6,
            &mut (),
            &mut (),
        ));
        assert_approx_eq!(Float, PI / 2.0, newton_bisection(
            0.0,
            2.0,
            |x, _, _| -> (Float, Float) {
                (x.cos(), -x.sin())
            },
            1e-6,
            1e-6,
            &mut (),
            &mut (),
        ));
        assert!(1e-5 > Float::abs(PI / 2.0 - newton_bisection(
            0.0,
            2.0,
            |x, _, _| -> (Float, Float) {
                (x.cos(), 10.0 * x.sin())
            },
            1e-6,
            1e-6,
            &mut (),
            &mut (),
        )));
        assert!(1e-6 > Float::abs(Float::sin(newton_bisection(
            0.1,
            10.1,
            |x, _, _| -> (Float, Float) {
                (x.sin(), x.cos())
            },
            1e-6,
            1e-6,
            &mut (),
            &mut (),
        ))));
        
        let f = |x: Float| -> (Float, Float) {
            (
                Float::powf(sqr(x.sin()), 0.05) - 0.3,
                0.1 * x.cos() * x.sin() / Float::powf(sqr(x.sin()), 0.95),
            )
        };
        assert!(1e-2 > Float::abs(f(newton_bisection(
            0.01,
            9.42477798,
            |x, _, _| f(x),
            1e-6,
            1e-6,
            &mut (),
            &mut (),
        )).0))
    }
}
