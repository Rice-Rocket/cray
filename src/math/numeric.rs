use crate::math_assert;
/// A trait that provides a blanket implementation for many traits a number
/// should have.

pub trait NumericNegative {
    const NEG_ONE: Self;
    const NEG_TWO: Self;

    fn nabs(self) -> Self;
    fn nsign(self) -> Self;
}

pub trait Numeric {
    const MIN: Self;
    const MAX: Self;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    fn nmin(self, rhs: Self) -> Self;
    fn nmax(self, rhs: Self) -> Self;
}

pub trait NumericFloat {
    const EPSILON: Self;
    const BIG_EPSILON: Self;
    const HALF: Self;

    fn nsqrt(self) -> Self;
    fn ninv(self) -> Self;
    fn nnan(self) -> bool;
    fn nfinite(self) -> bool;
    fn nacos(self) -> Self;
    fn npowf(self, n: Self) -> Self;
    fn nexp(self) -> Self;
    fn nround(self) -> Self;
    fn nfloor(self) -> Self;
    fn nceil(self) -> Self;
    fn ntrunc(self) -> Self;
}


macro_rules! impl_numeric {
    ($($ty:ident, $vmin:expr, $vmax:expr);* $(;)*) => {
        $(
            impl Numeric for $ty {
                const MAX: $ty = $vmax;
                const MIN: $ty = $vmin;

                const ZERO: $ty = 0 as $ty;
                const ONE: $ty = 1 as $ty;
                const TWO: $ty = 2 as $ty;

                fn nmin(self, rhs: Self) -> Self {
                    self.min(rhs)
                }
                fn nmax(self, rhs: Self) -> Self {
                    self.max(rhs)
                }
            }
        )*
    };
}

macro_rules! impl_numeric_negative {
    ($($ty:ident, $vmin:expr, $vmax:expr);* $(;)*) => {
        $(
            impl NumericNegative for $ty {
                const NEG_ONE: $ty = -1 as $ty;
                const NEG_TWO: $ty = -2 as $ty;

                fn nabs(self) -> Self {
                    self.abs()
                }
                fn nsign(self) -> Self {
                    self.signum()
                }
            }
        )*
    };
}

macro_rules! impl_numeric_float {
    ($($ty:ident, $half:expr, $eps:expr, $beps:expr);* $(;)*) => {
        $(
            impl NumericFloat for $ty {
                const EPSILON: Self = $eps;
                const BIG_EPSILON: Self = $eps;
                const HALF: Self = $half;

                fn nsqrt(self) -> Self {
                    self.sqrt()
                }
                fn ninv(self) -> Self {
                    math_assert!(self != 0 as $ty);
                    self.recip()
                }
                fn nnan(self) -> bool {
                    self.is_nan()
                }
                fn nfinite(self) -> bool {
                    self.is_finite()
                }
                fn nacos(self) -> Self {
                    self.clamp(-1.0, 1.0).acos()
                }
                fn npowf(self, n: Self) -> Self {
                    self.powf(n)
                }
                fn nexp(self) -> Self {
                    self.exp()
                }
                fn nround(self) -> Self {
                    self.round()
                }
                fn nfloor(self) -> Self {
                    self.floor()
                }
                fn nceil(self) -> Self {
                    self.ceil()
                }
                fn ntrunc(self) -> Self {
                    self.trunc()
                }
            }
        )*
    }
}

impl_numeric!(
    f32, f32::MIN, f32::MAX; f64, f64::MIN, f64::MAX;
    i8, i8::MIN, i8::MAX; i16, i16::MIN, i16::MAX; i32, i32::MIN, i32::MAX; i64, i64::MIN, i64::MAX; isize, isize::MIN, isize::MAX;
    u8, u8::MIN, u8::MAX; u16, u16::MIN, u16::MAX; u32, u32::MIN, u32::MAX; u64, u64::MIN, u64::MAX; usize, usize::MIN, usize::MAX;
);

impl_numeric_negative!(
    f32, f32::MIN, f32::MAX; f64, f64::MIN, f64::MAX;
    i8, i8::MIN, i8::MAX; i16, i16::MIN, i16::MAX; i32, i32::MIN, i32::MAX; i64, i64::MIN, i64::MAX; isize, isize::MIN, isize::MAX;
);

impl_numeric_float!(
    f32, 0.5f32, f32::EPSILON, 2e-4f32; f64, 0.5f64, f64::EPSILON, 2e-4f64
);
