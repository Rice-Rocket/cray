#![allow(clippy::assign_op_pattern)]
use std::ops::{Index, IndexMut, Neg};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use numeric::{DifferenceOfProducts, HasNan};

use crate::math::*;


// Code adapted from https://github.com/jalberse/shimmer/blob/main/src/interval.rs


pub fn add_round_up(a: Float, b: Float) -> Float {
    next_float_up(a + b)
}

pub fn add_round_down(a: Float, b: Float) -> Float {
    next_float_down(a + b)
}

pub fn sub_round_up(a: Float, b: Float) -> Float {
    next_float_up(a - b)
}

pub fn sub_round_down(a: Float, b: Float) -> Float {
    next_float_down(a - b)
}

pub fn mul_round_up(a: Float, b: Float) -> Float {
    next_float_up(a * b)
}

pub fn mul_round_down(a: Float, b: Float) -> Float {
    next_float_down(a * b)
}

pub fn div_round_up(a: Float, b: Float) -> Float {
    next_float_up(a / b)
}

pub fn div_round_down(a: Float, b: Float) -> Float {
    next_float_down(a / b)
}

pub fn sqrt_round_up(a: Float) -> Float {
    next_float_up(a.sqrt())
}

pub fn sqrt_round_down(a: Float) -> Float {
    next_float_down(a.sqrt())
}

pub fn fma_round_up(a: Float, b: Float, c: Float) -> Float {
    next_float_up(Float::mul_add(a, b, c))
}

pub fn fma_round_down(a: Float, b: Float, c: Float) -> Float {
    next_float_down(Float::mul_add(a, b, c))
}


#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default)]
pub struct Interval {
    low: Float,
    high: Float,
}

pub trait FloatInterval {
    fn new_interval(low: Float, high: Float) -> Self;
    fn from_value_and_error(v: Float, err: Float) -> Self;
    fn width(&self) -> Float;
    fn to_scalar(self) -> Float;
}

impl Interval {
    pub const fn from_val(v: Float) -> Interval {
        Interval { low: v, high: v }
    }

    pub fn upper_bound(&self) -> Float {
        self.high
    }

    pub fn lower_bound(&self) -> Float {
        self.low
    }

    pub fn midpoint(&self) -> Float {
        (self.low + self.high) / 2.0
    }

    pub fn floor(&self) -> Float {
        Float::floor(self.low)
    }

    pub fn ceil(&self) -> Float {
        Float::ceil(self.high)
    }

    pub fn exactly(&self, v: Float) -> bool {
        self.low == v && self.high == v
    }

    /// Checks if v is within the range (inclusive)
    pub fn in_range(&self, v: Float) -> bool {
        v >= self.lower_bound() && v <= self.upper_bound()
    }

    /// true if the two intervals overlap, else false
    pub fn overlap(&self, other: &Interval) -> bool {
        self.lower_bound() <= other.upper_bound() && self.upper_bound() >= other.lower_bound()
    }

    /// This is NOT just shorthand for a * a.
    /// It sometimes is able to compute a tighter bound than would be found
    /// with that method.
    pub fn sqr(&self) -> Interval {
        let alow = Float::abs(self.lower_bound());
        let ahigh = Float::abs(self.upper_bound());
        let (alow, ahigh) = if alow > ahigh {
            (ahigh, alow)
        } else {
            (alow, ahigh)
        };
        if self.in_range(0.0) {
            return Interval {
                low: 0.0,
                high: mul_round_up(ahigh, ahigh),
            };
        }
        Interval {
            low: mul_round_down(alow, alow),
            high: mul_round_up(ahigh, ahigh),
        }
    }
}

impl FloatInterval for Interval {
    fn new_interval(low: Float, high: Float) -> Interval {
        let low_real = Float::min(low, high);
        let high_real = Float::max(low, high);
        Interval {
            low: low_real,
            high: high_real,
        }
    }

    fn from_value_and_error(v: Float, err: Float) -> Interval {
        let (low, high) = if err == 0.0 {
            (v, v)
        } else {
            let low = sub_round_down(v, err);
            let high = add_round_up(v, err);
            (low, high)
        };
        Interval { low, high }
    }

    fn width(&self) -> Float {
        self.high - self.low
    }

    fn to_scalar(self) -> Float {
        self.midpoint()
    }
}

impl DifferenceOfProducts for Interval {
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        let ab = [
            a.low * b.low,
            a.high * b.low,
            a.low * b.high,
            a.high * b.high,
        ];
        debug_assert!(!ab.contains(&Float::NAN));
        let ab_low = ab.iter().fold(Float::NAN, |a, &b| a.min(b));
        let ab_high = ab.iter().fold(Float::NAN, |a, &b| a.max(b));

        let ab_low_index = if ab_low == ab[0] {
            0
        } else if ab_low == ab[1] {
            1
        } else if ab_low == ab[2] {
            2
        } else {
            3
        };

        let ab_high_index = if ab_high == ab[0] {
            0
        } else if ab_high == ab[1] {
            1
        } else if ab_high == ab[2] {
            2
        } else {
            3
        };

        let cd = [
            a.low * b.low,
            a.high * b.low,
            a.low * b.high,
            a.high * b.high,
        ];
        debug_assert!(!cd.contains(&Float::NAN));
        let cd_low = cd.iter().fold(Float::NAN, |a, &b| a.min(b));
        let cd_high = cd.iter().fold(Float::NAN, |a, &b| a.max(b));

        let cd_low_index = if cd_low == cd[0] {
            0
        } else if cd_low == cd[1] {
            1
        } else if cd_low == cd[2] {
            2
        } else {
            3
        };

        let cd_high_index = if cd_high == cd[0] {
            0
        } else if cd_high == cd[1] {
            1
        } else if cd_high == cd[2] {
            2
        } else {
            3
        };

        let low = Float::difference_of_products(
            a[ab_low_index & 1],
            b[ab_low_index >> 1],
            c[cd_high_index & 1],
            d[cd_high_index >> 1],
        );

        let high = Float::difference_of_products(
            a[ab_high_index & 1],
            b[ab_high_index >> 2],
            c[cd_low_index & 1],
            d[cd_low_index >> 1],
        );

        debug_assert!(low < high);

        Interval {
            low: next_float_down(next_float_down(low)),
            high: next_float_up(next_float_up(high)),
        }
    }

    fn sum_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        Self::difference_of_products(a, b, -c, d)
    }
}


impl NumericNegative for Interval {
    const NEG_ONE: Self = Interval::from_val(-1.0);
    const NEG_TWO: Self = Interval::from_val(-2.0);

    fn abs(self) -> Self {
        if self.low >= 0.0 {
            // The entire interval is greater than zero, so we're set
            self
        } else if self.high <= 0.0 {
            // The entire interval is less than zero
            Interval {
                low: -self.high,
                high: -self.low,
            }
        } else {
            // The interval straddles zero
            Interval::new_interval(0.0, Float::max(-self.low, self.high))
        }
    }

    fn sign(self) -> Self {
        if self.low >= 0.0 {
            Interval::from_val(1.0)
        } else if self.high <= 0.0 {
            Interval::from_val(-1.0)
        } else {
            Interval::from_val(1.0)
        }
    }
}

impl HasNan for Interval {
    const NAN: Self = Interval { low: Float::NAN, high: Float::NAN };

    fn has_nan(&self) -> bool {
        self.low.is_nan() || self.high.is_nan()
    }

    fn has_finite(&self) -> bool {
        self.low.is_finite() && self.high.is_finite()
    }
}

impl NumericFloat for Interval {
    const EPSILON: Self = Interval::from_val(Float::EPSILON);
    const BIG_EPSILON: Self = Interval::from_val(2e-4 as Float);
    const HALF: Self = Interval::from_val(0.5);

    fn nsqrt(self) -> Interval {
        Interval {
            low: sqrt_round_down(self.low),
            high: sqrt_round_up(self.high),
        }
    }

    fn ninv(self) -> Self {
        1.0 / self
    }

    fn nround(self) -> Self {
        Float::round(self.midpoint()).into()
    }

    fn ntrunc(self) -> Self {
        Float::round(self.midpoint()).into()
    }

    fn nfloor(self) -> Self {
        Float::floor(self.low).into()
    }

    fn nceil(self) -> Self {
        Float::ceil(self.high).into()
    }

    fn nacos(self) -> Self {
        Float::acos(self.midpoint()).into()
    }

    fn npowf(self, n: Self) -> Self {
        Float::powf(self.midpoint(), n.into()).into()
    }

    fn nexp(self) -> Self {
        Float::exp(self.midpoint()).into()
    }
}

impl NumericConsts for Interval {
    const MIN: Self = Interval::from_val(Float::MIN);
    const MAX: Self = Interval::from_val(Float::MAX);
    const ZERO: Self = Interval::from_val(0.0);
    const ONE: Self = Interval::from_val(1.0);
    const TWO: Self = Interval::from_val(2.0);
}

impl NumericOrd for Interval {
    fn nmin(self, rhs: Interval) -> Self {
        Float::min(self.low, rhs.low).into()
    }

    fn nmax(self, rhs: Self) -> Self {
        Float::max(self.high, rhs.high).into()
    }
}

impl From<Float> for Interval {
    fn from(value: Float) -> Self {
        Self::from_val(value)
    }
}

impl Index<usize> for Interval {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &self.low
        } else {
            &self.high
        }
    }
}

impl IndexMut<usize> for Interval {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &mut self.low
        } else {
            &mut self.high
        }
    }
}

impl Neg for Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        Interval {
            low: -self.high,
            high: -self.low,
        }
    }
}

impl Neg for &Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        Interval {
            low: -self.high,
            high: -self.low,
        }
    }
}

// Approximates the interval with a single float value, at its midpoint.
impl From<Interval> for Float {
    fn from(value: Interval) -> Self {
        value.midpoint()
    }
}

impl PartialEq<Float> for Interval {
    fn eq(&self, other: &Float) -> bool {
        self.exactly(*other)
    }

    // Note that != is not just negating the == implementation under interval arithmetic.
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Float) -> bool {
        *other < self.low || *other > self.high
    }
}

impl_op_ex!(+|a: &Interval, b: &Interval| -> Interval {
    Interval{ low: add_round_down(a.low, b.low), high: add_round_up(a.high, b.high) }
});

impl_op_ex!(+=|a: &mut Interval, b: &Interval|
{
    *a = *a - *b;
});

impl_op_ex!(-|a: &Interval, b: &Interval| -> Interval {
    Interval {
        low: sub_round_down(a.low, b.low),
        high: sub_round_up(a.high, b.high),
    }
});

impl_op_ex!(-=|a: &mut Interval, b: &Interval|
{
    *a = *a - *b;
});

impl_op_ex!(*|a: &Interval, b: &Interval| -> Interval {
    let lp: [Float; 4] = [
        mul_round_down(a.low, b.low),
        mul_round_down(a.high, b.low),
        mul_round_down(a.low, b.high),
        mul_round_down(a.high, b.high),
    ];
    let hp: [Float; 4] = [
        mul_round_up(a.low, b.low),
        mul_round_up(a.high, b.low),
        mul_round_up(a.low, b.high),
        mul_round_up(a.high, b.high),
    ];
    debug_assert!(!lp.contains(&Float::NAN));
    debug_assert!(!hp.contains(&Float::NAN));
    let low = lp.iter().fold(Float::NAN, |a, &b| a.min(b));
    let high = hp.iter().fold(Float::NAN, |a, &b| a.max(b));
    Interval { low, high }
});

impl_op_ex!(*=|a: &mut Interval, b: &Interval|
{
    *a = *a * *b;
});

impl_op_ex!(/|a: &Interval, b: &Interval| -> Interval
{
    if b.in_range(0.0)
    {
        // The interval we're dividing by straddles zero, so just return
        // the interval with everything
        return Interval{
            low: Float::NEG_INFINITY,
            high: Float::INFINITY
        }
    }
    let low_quot: [Float; 4]  = [
        div_round_down(a.low, b.low),
        div_round_down(a.high, b.low),
        div_round_down(a.low, b.high),
        div_round_down(a.high, b.high),
    ];
    let high_quot: [Float; 4]  = [
        div_round_up(a.low, b.low),
        div_round_up(a.high, b.low),
        div_round_up(a.low, b.high),
        div_round_up(a.high, b.high),
    ];
    let low = low_quot.iter().fold(Float::NAN, |a, &b| a.min(b));
    let high = high_quot.iter().fold(Float::NAN, |a, &b| a.max(b));
    Interval { low, high }
});

impl_op_ex!(/=|a: &mut Interval, b: &Interval|
{
    *a = *a / *b;
});

impl_op_ex_commutative!(+|a: &Interval, f: &Float| -> Interval
{
    a + Interval::from_val(*f)
});

impl_op_ex!(+=|a: &mut Interval, f: &Float|
{
    *a = *a + f;
});

impl_op_ex!(-|a: &Interval, f: &Float| -> Interval { a - Interval::from_val(*f) });

impl_op_ex!(-=|a: &mut Interval, f: &Float|
{
    *a = *a - f;
});

impl_op_ex!(-|f: Float, i: &Interval| -> Interval { Interval::from_val(f) - i });

impl_op_ex_commutative!(*|a: &Interval, f: &Float| -> Interval {
    if *f > 0.0 {
        Interval::new_interval(mul_round_down(*f, a.low), mul_round_up(*f, a.high))
    } else {
        Interval::new_interval(mul_round_down(*f, a.high), mul_round_up(*f, a.low))
    }
});

impl_op_ex!(*=|a: &mut Interval, f: &Float|
{
    *a = *a * f;
});

impl_op_ex!(/|a: &Interval, f: &Float| -> Interval {
    if *f > 0.0 {
        Interval::new_interval(div_round_down(a.low, *f), div_round_up(a.high, *f))
    } else {
        Interval::new_interval(div_round_down(a.high, *f), div_round_up(a.low, *f))
    }
});

impl_op_ex!(/=|a: &mut Interval, f: &Float|
{
    *a = *a / f;
});

impl_op_ex!(/|f: &Float, i: &Interval| -> Interval
{
    if i.in_range(0.0)
    {
        return Interval{ low: Float::NEG_INFINITY, high: Float::INFINITY };
    }
    if *f > 0.0
    {
        Interval::new_interval(div_round_down(*f, i.upper_bound()), div_round_up(*f, i.lower_bound()))
    }
    else {
        Interval::new_interval(div_round_down(*f, i.lower_bound()), div_round_up(*f, i.upper_bound()))
    }
});
