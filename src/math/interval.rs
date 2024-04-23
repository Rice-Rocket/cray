#![allow(clippy::assign_op_pattern)]
use std::ops::{Index, IndexMut, Neg};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};

use crate::math::{*, math::{next_float_up, next_float_down}};


// Code adapted from https://github.com/jalberse/shimmer/blob/main/src/interval.rs


pub fn add_round_up(a: Scalar, b: Scalar) -> Scalar {
    next_float_up(a + b)
}

pub fn add_round_down(a: Scalar, b: Scalar) -> Scalar {
    next_float_down(a + b)
}

pub fn sub_round_up(a: Scalar, b: Scalar) -> Scalar {
    next_float_up(a - b)
}

pub fn sub_round_down(a: Scalar, b: Scalar) -> Scalar {
    next_float_down(a - b)
}

pub fn mul_round_up(a: Scalar, b: Scalar) -> Scalar {
    next_float_up(a * b)
}

pub fn mul_round_down(a: Scalar, b: Scalar) -> Scalar {
    next_float_down(a * b)
}

pub fn div_round_up(a: Scalar, b: Scalar) -> Scalar {
    next_float_up(a / b)
}

pub fn div_round_down(a: Scalar, b: Scalar) -> Scalar {
    next_float_down(a / b)
}

pub fn sqrt_round_up(a: Scalar) -> Scalar {
    next_float_up(a.sqrt())
}

pub fn sqrt_round_down(a: Scalar) -> Scalar {
    next_float_down(a.sqrt())
}

pub fn fma_round_up(a: Scalar, b: Scalar, c: Scalar) -> Scalar {
    next_float_up(Scalar::mul_add(a, b, c))
}

pub fn fma_round_down(a: Scalar, b: Scalar, c: Scalar) -> Scalar {
    next_float_down(Scalar::mul_add(a, b, c))
}


#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Default)]
pub struct Interval {
    low: Scalar,
    high: Scalar,
}

pub trait FloatInterval {
    fn new_interval(low: Scalar, high: Scalar) -> Self;
    fn from_value_and_error(v: Scalar, err: Scalar) -> Self;
    fn width(&self) -> Scalar;
    fn to_scalar(self) -> Scalar;
}

impl Interval {
    pub const fn from_val(v: Scalar) -> Interval {
        Interval { low: v, high: v }
    }

    pub fn upper_bound(&self) -> Scalar {
        self.high
    }

    pub fn lower_bound(&self) -> Scalar {
        self.low
    }

    pub fn midpoint(&self) -> Scalar {
        (self.low + self.high) / 2.0
    }

    pub fn floor(&self) -> Scalar {
        Scalar::floor(self.low)
    }

    pub fn ceil(&self) -> Scalar {
        Scalar::ceil(self.high)
    }

    pub fn exactly(&self, v: Scalar) -> bool {
        self.low == v && self.high == v
    }

    /// Checks if v is within the range (inclusive)
    pub fn in_range(&self, v: Scalar) -> bool {
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
        let alow = Scalar::abs(self.lower_bound());
        let ahigh = Scalar::abs(self.upper_bound());
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
    fn new_interval(low: Scalar, high: Scalar) -> Interval {
        let low_real = Scalar::min(low, high);
        let high_real = Scalar::max(low, high);
        Interval {
            low: low_real,
            high: high_real,
        }
    }

    fn from_value_and_error(v: Scalar, err: Scalar) -> Interval {
        let (low, high) = if err == 0.0 {
            (v, v)
        } else {
            let low = sub_round_down(v, err);
            let high = add_round_up(v, err);
            (low, high)
        };
        Interval { low, high }
    }

    fn width(&self) -> Scalar {
        self.high - self.low
    }

    fn to_scalar(self) -> Scalar {
        self.midpoint()
    }
}


impl NumericNegative for Interval {
    const NEG_ONE: Self = Interval::from_val(-1.0);
    const NEG_TWO: Self = Interval::from_val(-2.0);

    fn nabs(self) -> Self {
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
            Interval::new_interval(0.0, Scalar::max(-self.low, self.high))
        }
    }

    fn nsign(self) -> Self {
        if self.low >= 0.0 {
            Interval::from_val(1.0)
        } else if self.high <= 0.0 {
            Interval::from_val(-1.0)
        } else {
            Interval::from_val(1.0)
        }
    }
}

impl NumericFloat for Interval {
    const EPSILON: Self = Interval::from_val(Scalar::EPSILON);
    const BIG_EPSILON: Self = Interval::from_val(2e-4f32);
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

    fn nnan(self) -> bool {
        self.low.is_nan() || self.high.is_nan()
    }

    fn nfinite(self) -> bool {
        self.low.is_finite() && self.high.is_finite()
    }

    fn nround(self) -> Self {
        Scalar::round(self.midpoint()).into()
    }

    fn ntrunc(self) -> Self {
        Scalar::round(self.midpoint()).into()
    }

    fn nfloor(self) -> Self {
        Scalar::floor(self.low).into()
    }

    fn nceil(self) -> Self {
        Scalar::ceil(self.high).into()
    }

    fn nacos(self) -> Self {
        Scalar::acos(self.midpoint()).into()
    }

    fn npowf(self, n: Self) -> Self {
        Scalar::powf(self.midpoint(), n.into()).into()
    }

    fn nexp(self) -> Self {
        Scalar::exp(self.midpoint()).into()
    }
}

impl Numeric for Interval {
    const MIN: Self = Interval::from_val(Scalar::MIN);
    const MAX: Self = Interval::from_val(Scalar::MAX);
    const ZERO: Self = Interval::from_val(0.0);
    const ONE: Self = Interval::from_val(1.0);
    const TWO: Self = Interval::from_val(2.0);

    fn nmin(self, rhs: Interval) -> Self {
        Scalar::min(self.low, rhs.low).into()
    }

    fn nmax(self, rhs: Self) -> Self {
        Scalar::max(self.high, rhs.high).into()
    }
}

impl From<Scalar> for Interval {
    fn from(value: Scalar) -> Self {
        Self::from_val(value)
    }
}

impl Index<usize> for Interval {
    type Output = Scalar;

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
impl From<Interval> for Scalar {
    fn from(value: Interval) -> Self {
        value.midpoint()
    }
}

impl PartialEq<Scalar> for Interval {
    fn eq(&self, other: &Scalar) -> bool {
        self.exactly(*other)
    }

    // Note that != is not just negating the == implementation under interval arithmetic.
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, other: &Scalar) -> bool {
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
    let lp: [Scalar; 4] = [
        mul_round_down(a.low, b.low),
        mul_round_down(a.high, b.low),
        mul_round_down(a.low, b.high),
        mul_round_down(a.high, b.high),
    ];
    let hp: [Scalar; 4] = [
        mul_round_up(a.low, b.low),
        mul_round_up(a.high, b.low),
        mul_round_up(a.low, b.high),
        mul_round_up(a.high, b.high),
    ];
    debug_assert!(!lp.contains(&Scalar::NAN));
    debug_assert!(!hp.contains(&Scalar::NAN));
    let low = lp.iter().fold(Scalar::NAN, |a, &b| a.min(b));
    let high = hp.iter().fold(Scalar::NAN, |a, &b| a.max(b));
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
            low: Scalar::NEG_INFINITY,
            high: Scalar::INFINITY
        }
    }
    let low_quot: [Scalar; 4]  = [
        div_round_down(a.low, b.low),
        div_round_down(a.high, b.low),
        div_round_down(a.low, b.high),
        div_round_down(a.high, b.high),
    ];
    let high_quot: [Scalar; 4]  = [
        div_round_up(a.low, b.low),
        div_round_up(a.high, b.low),
        div_round_up(a.low, b.high),
        div_round_up(a.high, b.high),
    ];
    let low = low_quot.iter().fold(Scalar::NAN, |a, &b| a.min(b));
    let high = high_quot.iter().fold(Scalar::NAN, |a, &b| a.max(b));
    Interval { low, high }
});

impl_op_ex!(/=|a: &mut Interval, b: &Interval|
{
    *a = *a / *b;
});

impl_op_ex_commutative!(+|a: &Interval, f: &Scalar| -> Interval
{
    a + Interval::from_val(*f)
});

impl_op_ex!(+=|a: &mut Interval, f: &Scalar|
{
    *a = *a + f;
});

impl_op_ex!(-|a: &Interval, f: &Scalar| -> Interval { a - Interval::from_val(*f) });

impl_op_ex!(-=|a: &mut Interval, f: &Scalar|
{
    *a = *a - f;
});

impl_op_ex!(-|f: Scalar, i: &Interval| -> Interval { Interval::from_val(f) - i });

impl_op_ex_commutative!(*|a: &Interval, f: &Scalar| -> Interval {
    if *f > 0.0 {
        Interval::new_interval(mul_round_down(*f, a.low), mul_round_up(*f, a.high))
    } else {
        Interval::new_interval(mul_round_down(*f, a.high), mul_round_up(*f, a.low))
    }
});

impl_op_ex!(*=|a: &mut Interval, f: &Scalar|
{
    *a = *a * f;
});

impl_op_ex!(/|a: &Interval, f: &Scalar| -> Interval {
    if *f > 0.0 {
        Interval::new_interval(div_round_down(a.low, *f), div_round_up(a.high, *f))
    } else {
        Interval::new_interval(div_round_down(a.high, *f), div_round_up(a.low, *f))
    }
});

impl_op_ex!(/=|a: &mut Interval, f: &Scalar|
{
    *a = *a / f;
});

impl_op_ex!(/|f: &Scalar, i: &Interval| -> Interval
{
    if i.in_range(0.0)
    {
        return Interval{ low: Scalar::NEG_INFINITY, high: Scalar::INFINITY };
    }
    if *f > 0.0
    {
        Interval::new_interval(div_round_down(*f, i.upper_bound()), div_round_up(*f, i.lower_bound()))
    }
    else {
        Interval::new_interval(div_round_down(*f, i.lower_bound()), div_round_up(*f, i.upper_bound()))
    }
});
