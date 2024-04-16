use super::na;

/// A trait that provides a blanket implementation for many traits a number
/// should have.
pub trait Numeric<T>: NumericRealField + na::Scalar + na::Field + na::ComplexField<RealField = T> {}
impl<T: NumericRealField + na::Scalar + na::Field + na::ComplexField<RealField = T>> Numeric<T> for T {}


pub trait NumericRealField {
    const MIN: Self;
    const MAX: Self;

    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;

    fn mini(self, rhs: Self) -> Self;
    fn maxi(self, rhs: Self) -> Self;
}

macro_rules! impl_numeric {
    ($($ty:ident, $vmin:expr, $vmax:expr);* $(;)*) => {
        $(
            impl NumericRealField for $ty {
                const MAX: $ty = $vmax;
                const MIN: $ty = $vmin;

                const ZERO: $ty = 0 as $ty;
                const ONE: $ty = 1 as $ty;
                const TWO: $ty = 2 as $ty;

                fn mini(self, rhs: Self) -> Self {
                    self.min(rhs)
                }
                fn maxi(self, rhs: Self) -> Self {
                    self.max(rhs)
                }
            }
        )*
    };
}

impl_numeric!(
    f32, f32::MIN, f32::MAX; f64, f64::MIN, f64::MAX;
    i8, i8::MIN, i8::MAX; i16, i16::MIN, i16::MAX; i32, i32::MIN, i32::MAX; i64, i64::MIN, i64::MAX; isize, isize::MIN, isize::MAX;
    u8, u8::MIN, u8::MAX; u16, u16::MIN, u16::MAX; u32, u32::MIN, u32::MAX; u64, u64::MIN, u64::MAX; usize, usize::MIN, usize::MAX;
);
