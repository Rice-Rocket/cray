use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};

use crate::math::*;


macro_rules! decl_vect {
    ($name:ident; $($v:ident),*) => {
        #[derive(Clone, Copy)]
        pub struct $name<T> {$(
            pub $v: T,
        )*}
    }
}


macro_rules! create_vect {
    ($name:ident; $a:ident, $($b:ident),*; $alias0:ident:$out0:ident, $($alias:ident:$out:ident:$x:ident-$($y:ident)-*),*) => {
        create_vect!($name; $a, $($b),*; $a; $($b),*; $alias0:$out0:$a-$($b)-*, $($alias:$out:$x-$($y)-*),*; $($alias:$x-$($y)-*),*);
    };
    ($name:ident; $($v:ident),*; $x:ident; $($y:ident),*; $($alias:ident:$out:ident:$($z:ident)-*),*; $($alias0:ident:$($z0:ident)-*),*) => {
        impl<T: Clone + Copy> $name<T> {
            #[inline]
            pub const fn new($($v: T,)*) -> Self {
                Self {$(
                    $v,
                )*}
            }

            #[inline]
            pub const fn splat(v: T) -> Self {
                Self {$(
                    $v: v,
                )*}
            }

            #[inline]
            pub fn map<U, F: Fn(T) -> U>(self, f: F) -> $name<U> {
                $name::<U> {$(
                    $v: f(self.$v),
                )*}
            }

            #[inline]
            pub fn zip<U>(self, rhs: $name<U>) -> $name<(T, U)> {
                $name::<(T, U)> {$(
                    $v: (self.$v, rhs.$v),
                )*}
            }
        }

        impl<T: Clone + Copy + Numeric + PartialOrd> $name<T> {
            pub const MIN: Self = Self::splat(T::MIN);
            pub const MAX: Self = Self::splat(T::MAX);
            pub const ZERO: Self = Self::splat(T::ZERO);
            pub const ONE: Self = Self::splat(T::ONE);

            #[inline]
            pub fn min_element(self) -> T {
                self.$x$(
                    .nmin(self.$y)
                )*
            }

            #[inline]
            pub fn max_element(self) -> T {
                self.$x$(
                    .nmax(self.$y)
                )*
            }

            #[inline]
            pub fn min(self, rhs: Self) -> Self {
                Self {$(
                    $v: self.$v.nmin(rhs.$v),
                )*}
            }

            #[inline]
            pub fn max(self, rhs: Self) -> Self {
                Self {$(
                    $v: self.$v.nmax(rhs.$v),
                )*}
            }

            #[inline]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                $(debug_assert!(min.$v < max.$v);)*
                self.max(min).min(max)
            }
        }

        impl<T: NumericField> $name<T> {
            #[inline]
            pub fn lerp(self, rhs: Self, t: T) -> Self {
                self + (rhs - self) * t
            }
        }

        impl<T: Numeric + NumericNegative + Mul<T, Output = T>> $name<T> {
            #[inline]
            pub fn abs(self) -> Self {
                Self {$(
                    $v: self.$v.nabs(),
                )*}
            }

            #[inline]
            pub fn signum(self) -> Self {
                Self {$(
                    $v: self.$v.nsign(),
                )*}
            }

            #[inline]
            pub fn copysign(self, rhs: Self) -> Self {
                self.abs() * rhs.signum()
            }
        }

        impl<T: Clone + Copy + Add<T, Output = T> + Mul<T, Output = T>> $name<T> {
            #[inline]
            pub fn dot(self, rhs: Self) -> T {
                self.$x * rhs.$x $(+ self.$y * rhs.$y)*
            }
        }

        impl<T: NumericField + NumericFloat + NumericNegative> $name<T> {
            #[inline]
            pub fn length_squared(self) -> T {
                self.dot(self)
            }

            #[inline]
            pub fn length(self) -> T {
                self.length_squared().nsqrt()
            }

            #[inline]
            pub fn length_squared_recip(self) -> T {
                self.length_squared().ninv()
            }

            #[inline]
            pub fn length_recip(self) -> T {
                self.length().ninv()
            }

            #[inline]
            pub fn normalize(self) -> Self {
                self * self.length_recip()
            }

            #[inline]
            pub fn is_normalized(self) -> bool {
                (self.length_squared() - T::ONE).nabs() <= T::BIG_EPSILON
            }

            #[inline]
            pub fn is_nan(self) -> bool {
                self.$x.has_nan() $(
                    || self.$y.has_nan()
                )*
            }

            #[inline]
            pub fn is_finite(self) -> bool {
                self.$x.has_finite() $(
                    && self.$y.has_finite()
                )*
            }

            #[inline]
            pub fn project_onto(self, rhs: Self) -> Self {
                rhs * self.dot(rhs) * rhs.length_squared_recip()
            }

            #[inline]
            pub fn reject_from(self, rhs: Self) -> Self {
                self - self.project_onto(rhs)
            }

            #[inline]
            pub fn distance(self, rhs: Self) -> T {
                (self - rhs).length()
            }

            #[inline]
            pub fn distance_squared(self, rhs: Self) -> T {
                (self - rhs).length_squared()
            }

            #[inline]
            pub fn round(self) -> Self {
                Self {$(
                    $v: self.$v.nround(),
                )*}
            }

            #[inline]
            pub fn floor(self) -> Self {
                Self {$(
                    $v: self.$v.nfloor(),
                )*}
            }

            #[inline]
            pub fn ceil(self) -> Self {
                Self {$(
                    $v: self.$v.nceil(),
                )*}
            }

            #[inline]
            pub fn trunc(self) -> Self {
                Self {$(
                    $v: self.$v.ntrunc(),
                )*}
            }

            #[inline]
            pub fn fract(self) -> Self {
                self - self.trunc()
            }

            #[inline]
            pub fn fract_floor(self) -> Self {
                self - self.floor()
            }

            #[inline]
            pub fn exp(self) -> Self {
                Self {$(
                    $v: self.$v.nexp(),
                )*}
            }

            #[inline]
            pub fn powf(self, n: T) -> Self {
                Self {$(
                    $v: self.$v.npowf(n),
                )*}
            }

            #[inline]
            pub fn recip(self) -> Self {
                Self {$(
                    $v: self.$v.ninv(),
                )*}
            }

            #[inline]
            pub fn midpoint(self, rhs: Self) -> Self {
                (self + rhs) * T::HALF
            }

            #[inline]
            pub fn angle(self, rhs: Self) -> T {
                (self.dot(rhs) / (self.length_squared() * rhs.length_squared()).nsqrt()).nacos()
            }

            #[inline]
            pub fn facing(self, rhs: Self) -> Self {
                if self.dot(rhs).nsign() < T::ZERO {
                    -self
                } else {
                    self
                }
            }
        }

        impl<T: Default> Default for $name<T> {
            #[inline]
            fn default() -> Self {
                Self {$(
                    $v: T::default(),
                )*}
            }
        }

        impl<T: Neg<Output = T>> Neg for $name<T> {
            type Output = $name<T>;
            fn neg(self) -> Self::Output {
                $name {$(
                    $v: -self.$v,
                )*}
            }
        }

        impl<T: Clone + Copy + Mul<T, Output = T>> Mul<T> for $name<T> {
            type Output = $name<T>;
            fn mul(self, rhs: T) -> Self::Output {
                $name {$(
                    $v: self.$v * rhs,
                )*}
            }
        }

        impl<T: Clone + Copy + Div<T, Output = T> + Numeric + PartialEq> Div<T> for $name<T> {
            type Output = $name<T>;
            fn div(self, rhs: T) -> Self::Output {
                debug_assert!(rhs != T::ZERO);
                $name {$(
                    $v: self.$v / rhs,
                )*}
            }
        }

        $(
            impl<T: Add<T, Output = T>> Add<$alias<T>> for $name<T> {
                type Output = $out<T>;
                fn add(self, rhs: $alias<T>) -> Self::Output {
                    $out {$(
                        $z: self.$z + rhs.$z,
                    )*}
                }
            }

            impl<T: Sub<T, Output = T>> Sub<$alias<T>> for $name<T> {
                type Output = $out<T>;
                fn sub(self, rhs: $alias<T>) -> Self::Output {
                    $out {$(
                        $z: self.$z - rhs.$z,
                    )*}
                }
            }

            impl<T: Mul<T, Output = T>> Mul<$alias<T>> for $name<T> {
                type Output = $out<T>;
                fn mul(self, rhs: $alias<T>) -> Self::Output {
                    $out {$(
                        $z: self.$z * rhs.$z,
                    )*}
                }
            }

            impl<T: Div<T, Output = T> + Numeric + PartialEq> Div<$alias<T>> for $name<T> {
                type Output = $out<T>;
                fn div(self, rhs: $alias<T>) -> Self::Output {
                    $(debug_assert!(rhs.$z != T::ZERO);)*
                    $out {$(
                        $z: self.$z / rhs.$z,
                    )*}
                }
            }

        )*
        $(
            impl<T> From<$alias0<T>> for $name<T> {
                fn from(value: $alias0<T>) -> Self {
                    Self {$(
                        $z0: value.$z0,
                    )*}
                }
            }
        )*

        impl<T: PartialEq> PartialEq for $name<T> {
            fn eq(&self, rhs: &Self) -> bool {
                self.$x == rhs.$x $(
                    && self.$y == rhs.$y
                )*
            }
        }
    }
}


decl_vect!(TPoint2; x,y);
decl_vect!(TPoint3; x,y,z);
decl_vect!(TPoint4; x,y,z,w);
decl_vect!(TVec2; x,y);
decl_vect!(TVec3; x,y,z);
decl_vect!(TVec4; x,y,z,w);
decl_vect!(TNormal2; x,y);
decl_vect!(TNormal3; x,y,z);

create_vect!(TPoint2; x, y; TPoint2:TPoint2, TVec2:TPoint2:x-y, TNormal2:TPoint2:x-y);
create_vect!(TPoint3; x,y,z; TPoint3:TPoint3, TVec3:TPoint3:x-y-z, TNormal3:TPoint3:x-y-z);
create_vect!(TPoint4; x,y,z,w; TPoint4:TPoint4, TVec4:TPoint4:x-y-z-w);
create_vect!(TVec2; x,y; TVec2:TVec2, TPoint2:TPoint2:x-y, TNormal2:TVec2:x-y);
create_vect!(TVec3; x,y,z; TVec3:TVec3, TPoint3:TPoint3:x-y-z, TNormal3:TVec3:x-y-z);
create_vect!(TVec4; x,y,z,w; TVec4:TVec4, TPoint4:TPoint4:x-y-z-w);
create_vect!(TNormal2; x,y; TNormal2:TNormal2, TPoint2:TPoint2:x-y, TVec2:TVec2:x-y);
create_vect!(TNormal3; x,y,z; TNormal3:TNormal3, TPoint3:TPoint3:x-y-z, TVec3:TVec3:x-y-z);


macro_rules! impl_index {
    ($($ty:ident; $dim:expr; $($index:expr => $itoken:tt => $v:ident),*);* $(;)*) => {
        $(
            impl<T> Index<usize> for $ty<T> {
                type Output = T;
                fn index(&self, index: usize) -> &Self::Output {
                    match index {
                        $(
                            $index => &self.$v,
                        )*
                        _ => panic!("cannot index into {} with {}", stringify!($ty), index)
                    }
                }
            }

            impl<T> IndexMut<usize> for $ty<T> {
                fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                    match index {
                        $(
                            $index => &mut self.$v,
                        )*
                        _ => panic!("cannot index into {} with {}", stringify!($ty), index)
                    }
                }
            }

            impl<T: Clone + Copy> From<[T; $dim]> for $ty<T> {
                #[inline]
                fn from(value: [T; $dim]) -> Self {
                    Self {$(
                        $v: value[$index],
                    )*}
                }
            }

            impl<T> From<$ty<T>> for [T; $dim] {
                #[inline]
                fn from(value: $ty<T>) -> [T; $dim] {
                    [$(value.$v,)*]
                }
            }

            impl<T> From<($(impl_index!($v),)*)> for $ty<T> {
                fn from(value: ($(impl_index!($v),)*)) -> Self {
                    Self {$(
                        $v: value.$itoken,
                    )*}
                }
            }

            impl<T> From<$ty<T>> for ($(impl_index!($v),)*) {
                fn from(value: $ty<T>) -> Self {
                    ($(value.$v,)*)
                }
            }
        )*
    };
    ($i:ident) => { T }
}


impl_index!(
    TPoint2; 2; 0 => 0 => x, 1 => 1 => y;
    TPoint3; 3; 0 => 0 => x, 1 => 1 => y, 2 => 2 => z;
    TPoint4; 4; 0 => 0 => x, 1 => 1 => y, 2 => 2 => z, 3 => 3 => w;
    TVec2; 2; 0 => 0 => x, 1 => 1 => y;
    TVec3; 3; 0 => 0 => x, 1 => 1 => y, 2 => 2 => z;
    TVec4; 4; 0 => 0 => x, 1 => 1 => y, 2 => 2 => z, 3 => 3 => w;
    TNormal2; 2; 0 => 0 => x, 1 => 1 => y;
    TNormal3; 3; 0 => 0 => x, 1 => 1 => y, 2 => 2 => z;
);


macro_rules! impl_cross {
    ($($ty:ident),*) => {
        $(
            impl<T: Clone + Copy + Sub<T, Output = T> + Mul<T, Output = T>> $ty<T> {
                #[inline]
                pub fn cross(self, rhs: Self) -> Self {
                    Self {
                        x: self.y * rhs.z - self.z * rhs.y,
                        y: self.z * rhs.x - self.x * rhs.z,
                        z: self.x * rhs.y - self.y * rhs.x
                    }
                }
            }
        )*
    }
}

impl_cross!(TVec3, TNormal3);


macro_rules! impl_interval {
    ($($ty:ident => $prim:ident: $($i:ident),*);* $(;)*) => {
        impl_interval!(@call $($ty => $prim: $($i),*; $($i),*);*);
    };
    (@call $($ty:ident => $prim:ident: $($i:ident),*; $a:ident, $($b:ident),*);*) => {
        $(
            impl<T: NumericField + NumericFloat + FloatInterval> $ty<T> {
                pub fn from_errors(v: $ty<$prim>, err: $ty<$prim>) -> Self {
                    Self::new($(
                        T::from_value_and_error(v.$i, err.$i),
                    )*)
                }

                pub fn error(self) -> $ty<$prim> {
                    $ty::<$prim>::new($(
                        self.$i.width() * $prim::HALF,
                    )*)
                }

                pub fn is_exact(self) -> bool {
                    self.$a.width() == $prim::ZERO $(
                        && self.$b.width() == $prim::ZERO
                    )*
                }
            }

            impl<T: NumericField + FloatInterval> From<$ty<$prim>> for $ty<T> {
                fn from(value: $ty<$prim>) -> Self {
                    Self::new($(
                        T::new_interval(value.$i, value.$i),
                    )*)
                }
            }

            impl<T: NumericField + FloatInterval> From<$ty<T>> for $ty<$prim> {
                fn from(value: $ty<T>) -> Self {
                    Self::new($(
                        value.$i.to_scalar(),
                    )*)
                }
            }
        )*
    };
}

impl_interval!(
    TPoint2 => Scalar: x,y;
    TPoint3 => Scalar: x,y,z;
    TPoint4 => Scalar: x,y,z,w;
    TVec2 => Scalar: x,y;
    TVec3 => Scalar: x,y,z;
    TVec4 => Scalar: x,y,z,w;
    TNormal2 => Scalar: x,y;
    TNormal3 => Scalar: x,y,z;
);


macro_rules! impl_debug {
    ($($name:ident: $($v:ident),*);* $(;)*) => {
        $(
            impl<T: Clone + Copy + std::fmt::Debug> std::fmt::Debug for $name<T> {
                fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(fmt, "{}{:?}", stringify!($name), <($(impl_debug!($v)),*)>::from(*self))
                }
            }
        )*
    };
    ($v:ident) => { T }
}

impl_debug!(
    TPoint2: x,y;
    TPoint3: x,y,z;
    TPoint4: x,y,z,w;
    TVec2: x,y;
    TVec3: x,y,z;
    TVec4: x,y,z,w;
    TNormal2: x,y;
    TNormal3: x,y,z;
);


macro_rules! impl_approx {
    ($($name:ident: $x:ident, $($y:ident),*);* $(;)*) => {
        $(
            impl<T: AbsDiffEq> AbsDiffEq for $name<T> where
                T::Epsilon: Copy,
            {
                type Epsilon = T::Epsilon;

                fn default_epsilon() -> T::Epsilon {
                    T::default_epsilon()
                }

                fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
                    T::abs_diff_eq(&self.$x, &other.$x, epsilon) $(
                        && T::abs_diff_eq(&self.$y, &other.$y, epsilon)
                    )*
                }
            }

            impl<T: RelativeEq> RelativeEq for $name<T> where
                T::Epsilon: Copy,
            {
                fn default_max_relative() -> T::Epsilon {
                    T::default_max_relative()
                }

                fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
                    T::relative_eq(&self.$x, &other.$x, epsilon, max_relative) $(
                        && T::relative_eq(&self.$y, &other.$y, epsilon, max_relative)
                    )*
                }
            }

            impl<T: UlpsEq> UlpsEq for $name<T> where
                T::Epsilon: Copy,
            {
                fn default_max_ulps() -> u32 {
                    T::default_max_ulps()
                }

                fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
                    T::ulps_eq(&self.$x, &other.$x, epsilon, max_ulps) $(
                        && T::ulps_eq(&self.$y, &other.$y, epsilon, max_ulps)
                    )*
                }
            }
        )*
    }
}

impl_approx!(
    TPoint2: x,y;
    TPoint3: x,y,z;
    TPoint4: x,y,z,w;
    TVec2: x,y;
    TVec3: x,y,z;
    TVec4: x,y,z,w;
    TNormal2: x,y;
    TNormal3: x,y,z;
);


impl<T: NumericField + NumericNegative + NumericFloat> TVec3<T> {
    pub fn local_basis(self) -> (TVec3<T>, TVec3<T>) {
        debug_assert!(self.is_normalized());

        let sign = self.z.nsign();
        let a = T::NEG_ONE / (sign + self.z);
        let b = self.x * self.y * a;
        (
            TVec3::new(T::ONE + sign * sqr(self.x) * a, sign * b, -sign * self.x),
            TVec3::new(b, sign + sqr(self.y) * a, -self.y)
        )
    }
}
