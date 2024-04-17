#![allow(dead_code)]

pub mod bounds;
pub mod dim;
pub mod numeric;
pub mod ray;
pub mod sphere;

pub use bounds::{direction::DirectionCone, Bounds2, Bounds3, IBounds2, IBounds3, UBounds2, UBounds3};
pub use dim::Dimension;
#[allow(unused_imports)]
pub use nalgebra::{self as na, Transform3, Unit};
#[allow(unused_imports)]
pub use nalgebra_glm::{self as glm, vec2, vec3, vec4};
pub use nalgebra_glm::{Qua, TMat2, TMat3, TMat4, TVec2, TVec3, TVec4};
pub use numeric::Numeric;
pub use ray::Ray;
pub use sphere::OctahedralVec3;


#[allow(clippy::module_inception)]
#[allow(clippy::excessive_precision)]
pub mod math {
    // Pbrt B.2 Mathematical Infrastructure

    pub use super::sphere::*;
    use super::ScalarAsBits;
    use crate::math::{Numeric, Scalar};

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

    /// Computes the smoothstep function at a given input `t` with bounds `a`
    /// and `b`.
    // TODO: Test this
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
    pub fn sqr<T: Numeric + Clone + Copy>(x: T) -> T {
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
    // TODO: Test this
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
    // TODO: Test this
    #[inline]
    pub fn bits_to_float(ui: ScalarAsBits) -> Scalar {
        let rf: Scalar;
        unsafe {
            let f: Scalar = std::mem::transmute_copy(&ui);
            rf = f;
        }
        rf
    }

    /// Returns the exponent of the bit representation of the float. (+127)
    // TODO: Test this
    #[inline]
    pub fn exponent(v: Scalar) -> ScalarAsBits {
        float_to_bits(v) >> 23
    }

    /// Returns the mantissa of the bit representation of the float.
    // TODO: Test this
    #[inline]
    pub fn significand(v: Scalar) -> ScalarAsBits {
        float_to_bits(v) & ((1 << 23) - 1)
    }

    /// Evaluates the polynomial given its coefficients.
    // TODO: Test this
    pub fn evaluate_polynomial(t: Scalar, c: Scalar, c_remaining: Option<&[Scalar]>) -> Scalar {
        if let Some(remain) = c_remaining {
            let eval = if remain.len() > 1 {
                evaluate_polynomial(t, remain[0], Some(&remain[1..]))
            } else {
                evaluate_polynomial(t, remain[0], None)
            };
            t * eval + c
        } else {
            c
        }
    }

    /// Computes the integer component of `log2(x)`.
    // TODO: Test this
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

    /// A fast e^x approximation using floating point bit magic.
    // TODO: Test this
    pub fn fast_exp(x: Scalar) -> Scalar {
        #[allow(clippy::approx_constant)]
        let xp = x * 1.442695041;

        let fxp = xp.floor();
        let f = xp - fxp;
        let i = fxp as ScalarAsBits;

        let pow_2_f = evaluate_polynomial(f, 1.0, Some(&[0.695556856, 0.226173572, 0.0781455737]));

        let exponent = exponent(pow_2_f) + i;
        if exponent < 1 {
            return 0.0;
        };
        if exponent > 254 {
            return Scalar::INFINITY;
        };

        let mut bits = float_to_bits(pow_2_f);
        bits &= 0b10000000011111111111111111111111;
        bits |= (exponent + 127) << 23;
        bits_to_float(bits)
    }

    const A1: Scalar = 0.254829592;
    const A2: Scalar = -0.284496736;
    const A3: Scalar = 1.421413741;
    const A4: Scalar = -1.453152027;
    const A5: Scalar = 1.061405429;
    const P: Scalar = 0.3275911;
    /// Computes the Gauss error function
    // TODO: Test this
    #[inline]
    pub fn erf(mut x: Scalar) -> Scalar {
        let sign = x.signum();
        x = x.abs();

        let t = 1.0 / (1.0 + P * x);
        let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

        sign * y
    }

    /// Computes the guassian distribution at the input `x`.
    ///
    /// Defaults for `mu` and `sigma` should be 0 and 1 respectively.
    // TODO: Test this
    #[inline]
    pub fn gaussian(x: Scalar, mu: Scalar, sigma: Scalar) -> Scalar {
        1.0 / (2.0 * PI * sigma * sigma).sqrt() * fast_exp(-sqr(x - mu) / (2.0 * sigma * sigma))
    }

    /// Computes the integral of the gaussian distribution at the input `x`.
    ///
    /// Defaults for `mu` and `sigma` should be 0 and 1 respectively.
    // TODO: Test this
    #[inline]
    pub fn gaussian_integral(x0: Scalar, x1: Scalar, mu: Scalar, sigma: Scalar) -> Scalar {
        let sigma_root_2 = sigma * SQRT_2;
        0.5 * (erf((mu - x0) / sigma_root_2) - erf((mu - x1) / sigma_root_2))
    }

    /// Computes the logistic distribution at the input `x` and a scale factor
    /// `s`.
    // TODO: Test this
    #[inline]
    pub fn logistic(mut x: Scalar, s: Scalar) -> Scalar {
        x = x.abs();
        (-x / s).exp() / (s * sqr(1.0 + (-x / s).exp()))
    }

    /// Computes the cumulative distribution function (CDF) of the logistic
    /// distribution at the input `x` and a scale factor `s`.
    // TODO: Test this
    #[inline]
    pub fn logistic_cdf(x: Scalar, s: Scalar) -> Scalar {
        1.0 / (1.0 + (-x / s).exp())
    }

    /// The logistic function limited to the interval `[a, b]` and renormalized.
    // TODO: Test this
    #[inline]
    pub fn trimmed_logistic(x: Scalar, s: Scalar, a: Scalar, b: Scalar) -> Scalar {
        logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
    }


    pub mod safe {
        use crate::math::{Numeric, Scalar};

        #[inline]
        pub fn sqrt<T: Numeric>(x: T) -> T {
            (x.maxi(T::ZERO)).sqrt()
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
}


pub type Scalar = f32;
pub type ScalarAsBits = u32;

pub type Quat = Qua<Scalar>;
pub type Mat2 = TMat2<Scalar>;
pub type Mat3 = TMat3<Scalar>;
pub type Mat4 = TMat4<Scalar>;
pub type Vec2 = TVec2<Scalar>;
pub type Vec3 = TVec3<Scalar>;
pub type Vec4 = TVec4<Scalar>;
