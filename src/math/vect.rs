use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg, Index, IndexMut};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};

use crate::math::*;
use crate::numeric::DifferenceOfProducts;

macro_rules! decl_vect {
    ($name:ident; $($v:ident),*) => {
        #[derive(Clone, Copy)]
        pub struct $name<T> {$(
            pub $v: T,
        )*}
    }
}

decl_vect!(Point2; x,y);
decl_vect!(Point3; x,y,z);
decl_vect!(Point4; x,y,z,w);
decl_vect!(Vec2; x,y);
decl_vect!(Vec3; x,y,z);
decl_vect!(Vec4; x,y,z,w);
decl_vect!(Normal2; x,y);
decl_vect!(Normal3; x,y,z);

macro_rules! impl_new {
    ($name:ident; $($v:ident),*) => {
        impl<T> $name<T> {
            #[inline]
            pub const fn new($($v: T,)*) -> Self {
                Self {$(
                    $v,
                )*}
            }
        }

        impl<T: Copy> $name<T> {
            #[inline]
            pub const fn splat(v: T) -> Self {
                Self {$(
                    $v: v,
                )*}
            }
        }
    }
}

impl_new!(Point2; x,y);
impl_new!(Point3; x,y,z);
impl_new!(Point4; x,y,z,w);
impl_new!(Vec2; x,y);
impl_new!(Vec3; x,y,z);
impl_new!(Vec4; x,y,z,w);
impl_new!(Normal2; x,y);
impl_new!(Normal3; x,y,z);

macro_rules! impl_map_zip {
    ($name:ident; $($v:ident),*) => {
        impl<T> $name<T> {
            #[inline]
            pub fn map<F, U>(self, f: F) -> $name<U>
            where
                F: Fn(T) -> U
            {
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
    }
}

impl_map_zip!(Point2; x,y);
impl_map_zip!(Point3; x,y,z);
impl_map_zip!(Point4; x,y,z,w);
impl_map_zip!(Vec2; x,y);
impl_map_zip!(Vec3; x,y,z);
impl_map_zip!(Vec4; x,y,z,w);
impl_map_zip!(Normal2; x,y);
impl_map_zip!(Normal3; x,y,z);

macro_rules! impl_consts {
    ($name2:ident, $name3:ident) => {
        impl<T: NumericConsts> $name2<T> {
            pub const MIN: Self = Self::new(T::MIN, T::MIN);
            pub const MAX: Self = Self::new(T::MAX, T::MAX);
            pub const ZERO: Self = Self::new(T::ZERO, T::ZERO);
            pub const ONE: Self = Self::new(T::ONE, T::ONE);
        }

        impl<T: NumericConsts> $name3<T> {
            pub const MIN: Self = Self::new(T::MIN, T::MIN, T::MIN);
            pub const MAX: Self = Self::new(T::MAX, T::MAX, T::MAX);
            pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO);
            pub const ONE: Self = Self::new(T::ONE, T::ONE, T::ONE);
        }
    }
}

macro_rules! impl_consts_4 {
    ($name:ident) => {
        impl<T: NumericConsts> $name<T> {
            pub const MIN: Self = Self::new(T::MIN, T::MIN, T::MIN, T::MIN);
            pub const MAX: Self = Self::new(T::MAX, T::MAX, T::MAX, T::MAX);
            pub const ZERO: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO);
            pub const ONE: Self = Self::new(T::ONE, T::ONE, T::ONE, T::ONE);
        }
    }
}

impl_consts!(Point2, Point3);
impl_consts!(Vec2, Vec3);
impl_consts!(Normal2, Normal3);
impl_consts_4!(Point4);
impl_consts_4!(Vec4);

pub trait Dot<Rhs> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

macro_rules! impl_binop {
    ($trait_name:ident, $fn_name:ident, [$self:ident: $ta:ident, $rhs:ident: $tb:ident] -> { $f:expr } where $($where_clause:tt)*) => {
        impl<T> $trait_name<$tb<T>> for $ta<T>
        where
            $($where_clause)*
        {
            type Output = T;

            fn $fn_name($self, $rhs: $tb<T>) -> Self::Output {
                $f
            }
        }
    };
    ($trait_name:ident, $fn_name:ident, [$self:ident: $ta:ident, $rhs:ident: $tb:ident] -> $output:ident { $f:expr } where $($where_clause:tt)*) => {
        impl<T> $trait_name<$tb<T>> for $ta<T>
        where
            $($where_clause)*
        {
            type Output = $output<T>;

            fn $fn_name($self, $rhs: $tb<T>) -> Self::Output {
                $f
            }
        }
    };
}

macro_rules! impl_binop_2 {
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> T { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Point2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Vec2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Normal2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Point2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Vec2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Normal2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Point2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Vec2] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Normal2] -> { $f } where $($where_clause)*);
    };
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> Self { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Point2] -> Point2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Vec2] -> Point2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point2, $rhs: Normal2] -> Point2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Point2] -> Vec2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Vec2] -> Vec2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec2, $rhs: Normal2] -> Vec2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Point2] -> Normal2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Vec2] -> Normal2 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal2, $rhs: Normal2] -> Normal2 { $f } where $($where_clause)*);
    };
}

macro_rules! impl_binop_3 {
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> T { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Point3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Vec3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Normal3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Point3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Vec3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Normal3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Point3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Vec3] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Normal3] -> { $f } where $($where_clause)*);
    };
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> Self { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Point3] -> Point3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Vec3] -> Point3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point3, $rhs: Normal3] -> Point3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Point3] -> Vec3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Vec3] -> Vec3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec3, $rhs: Normal3] -> Vec3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Point3] -> Normal3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Vec3] -> Normal3 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Normal3, $rhs: Normal3] -> Normal3 { $f } where $($where_clause)*);
    };
}

macro_rules! impl_binop_4 {
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> T { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point4, $rhs: Point4] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point4, $rhs: Vec4] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec4, $rhs: Point4] -> { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec4, $rhs: Vec4] -> { $f } where $($where_clause)*);
    };
    ($trait_name:ident, $fn_name:ident, [$self:ident, $rhs:ident] -> Self { $f:expr } where $($where_clause:tt)*) => {
        impl_binop!($trait_name, $fn_name, [$self: Point4, $rhs: Point4] -> Point4 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Point4, $rhs: Vec4] -> Point4 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec4, $rhs: Point4] -> Vec4 { $f } where $($where_clause)*);
        impl_binop!($trait_name, $fn_name, [$self: Vec4, $rhs: Vec4] -> Vec4 { $f } where $($where_clause)*);
    };
}

impl_binop_2!(Dot, dot, [self, rhs] -> T { self.x * rhs.x + self.y * rhs.y } where T: Mul<T, Output = T> + Add<T, Output = T>);
impl_binop_3!(Dot, dot, [self, rhs] -> T { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z } where T: Mul<T, Output = T> + Add<T, Output = T>);
impl_binop_4!(Dot, dot, [self, rhs] -> T { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w } where T: Mul<T, Output = T> + Add<T, Output = T>);

impl_binop_2!(Add, add, [self, rhs] -> Self { Self::new(self.x + rhs.x, self.y + rhs.y) } where T: Add<T, Output = T>);
impl_binop_3!(Add, add, [self, rhs] -> Self { Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) } where T: Add<T, Output = T>);
impl_binop_4!(Add, add, [self, rhs] -> Self { Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w) } where T: Add<T, Output = T>);

impl_binop_2!(Sub, sub, [self, rhs] -> Self { Self::new(self.x - rhs.x, self.y - rhs.y) } where T: Sub<T, Output = T>);
impl_binop_3!(Sub, sub, [self, rhs] -> Self { Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) } where T: Sub<T, Output = T>);
impl_binop_4!(Sub, sub, [self, rhs] -> Self { Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w) } where T: Sub<T, Output = T>);

impl_binop_2!(Mul, mul, [self, rhs] -> Self { Self::new(self.x * rhs.x, self.y * rhs.y) } where T: Mul<T, Output = T>);
impl_binop_3!(Mul, mul, [self, rhs] -> Self { Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z) } where T: Mul<T, Output = T>);
impl_binop_4!(Mul, mul, [self, rhs] -> Self { Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w) } where T: Mul<T, Output = T>);

impl_binop_2!(Div, div, [self, rhs] -> Self { Self::new(self.x / rhs.x, self.y / rhs.y) } where T: Div<T, Output = T>);
impl_binop_3!(Div, div, [self, rhs] -> Self { Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z) } where T: Div<T, Output = T>);
impl_binop_4!(Div, div, [self, rhs] -> Self { Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w) } where T: Div<T, Output = T>);

macro_rules! impl_vec_ops {
    ($name:ident; $a:ident, $($b:ident),*; $alias0:ident:$out0:ident, $($alias:ident:$out:ident:$x:ident-$($y:ident)-*),*) => {
        impl_vec_ops!($name; $a, $($b),*; $a; $($b),*; $alias0:$out0:$a-$($b)-*, $($alias:$out:$x-$($y)-*),*; $($alias:$x-$($y)-*),*);
    };
    ($name:ident; $($v:ident),*; $x:ident; $($y:ident),*; $($alias:ident:$out:ident:$($z:ident)-*),*; $($alias0:ident:$($z0:ident)-*),*) => {
        impl<T: NumericOrd> $name<T> {
            #[inline]
            pub fn min_element(self) -> T {
                self.$x$(.nmin(self.$y))*
            }

            #[inline]
            pub fn max_element(self) -> T {
                self.$x$(.nmax(self.$y))*
            }
        }

        impl<T: NumericOrd> $name<T> {
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
            pub fn clamp(self, min: Self, max: Self) -> Self
            where
                T: PartialOrd
            {
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

        impl<T: NumericNegative> $name<T> {
            #[inline]
            pub fn abs(self) -> Self {
                Self {$(
                    $v: self.$v.abs(),
                )*}
            }

            #[inline]
            pub fn signum(self) -> Self {
                Self {$(
                    $v: self.$v.sign(),
                )*}
            }

            #[inline]
            pub fn copysign(self, rhs: Self) -> Self
            where
                T: Mul<T, Output = T>
            {
                self.abs() * rhs.signum()
            }
        }

        impl<T: NumericField + NumericFloat + NumericNegative> $name<T> {
            #[inline]
            pub fn length_squared(self) -> T {
                self.dot(self)
            }

            #[inline]
            pub fn gram_schmidt(self, w: Self) -> Self {
                self - Self::splat(self.dot(w)) * w
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
                self / self.length()
            }

            #[inline]
            pub fn is_normalized(self) -> bool {
                (self.length_squared() - T::ONE).abs() <= T::BIG_EPSILON
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

            // NOTE: This may be wrong, but I think it should be fine
            #[inline]
            pub fn angle(self, rhs: Self) -> T {
                (self.dot(rhs) / (self.length_squared() * rhs.length_squared()).nsqrt()).nacos()
            }

            #[inline]
            pub fn facing(self, rhs: Self) -> Self {
                if self.dot(rhs).sign() < T::ZERO {
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

        impl<T: Clone + Copy + Div<T, Output = T> + NumericConsts + PartialEq> Div<T> for $name<T> {
            type Output = $name<T>;
            fn div(self, rhs: T) -> Self::Output {
                debug_assert!(rhs != T::ZERO);
                $name {$(
                    $v: self.$v / rhs,
                )*}
            }
        }

        impl<T: Clone + Copy + MulAssign<T>> MulAssign<T> for $name<T> {
            fn mul_assign(&mut self, rhs: T) {
                $(
                    self.$v *= rhs;
                )*
            }
        }

        impl<T: Clone + Copy + DivAssign<T>> DivAssign<T> for $name<T> {
            fn div_assign(&mut self, rhs: T) {
                $(
                    self.$v /= rhs;
                )*
            }
        }

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

impl_vec_ops!(Point2; x, y; Point2:Point2, Vec2:Point2:x-y, Normal2:Point2:x-y);
impl_vec_ops!(Point3; x,y,z; Point3:Point3, Vec3:Point3:x-y-z, Normal3:Point3:x-y-z);
impl_vec_ops!(Point4; x,y,z,w; Point4:Point4, Vec4:Point4:x-y-z-w);
impl_vec_ops!(Vec2; x,y; Vec2:Vec2, Point2:Point2:x-y, Normal2:Vec2:x-y);
impl_vec_ops!(Vec3; x,y,z; Vec3:Vec3, Point3:Point3:x-y-z, Normal3:Vec3:x-y-z);
impl_vec_ops!(Vec4; x,y,z,w; Vec4:Vec4, Point4:Point4:x-y-z-w);
impl_vec_ops!(Normal2; x,y; Normal2:Normal2, Point2:Point2:x-y, Vec2:Vec2:x-y);
impl_vec_ops!(Normal3; x,y,z; Normal3:Normal3, Point3:Point3:x-y-z, Vec3:Vec3:x-y-z);

macro_rules! impl_index {
    ($($ty:ident; $dim:expr; $($index:tt / $axis:ident => $v:ident),*);* $(;)*) => {
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

            impl<T> Index<Dimension> for $ty<T> {
                type Output = T;
                fn index(&self, index: Dimension) -> &Self::Output {
                    match index {
                        $(
                            Dimension::$axis => &self.$v,
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
                        $v: value.$index,
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
    Point2; 2; 0 / X => x, 1 / Y => y;
    Point3; 3; 0 / X => x, 1 / Y => y, 2 / Z => z;
    Point4; 4; 0 / X => x, 1 / Y => y, 2 / Z => z, 3 / W => w;
    Vec2; 2; 0 / X => x, 1 / Y => y;
    Vec3; 3; 0 / X => x, 1 / Y => y, 2 / Z => z;
    Vec4; 4; 0 / X => x, 1 / Y => y, 2 / Z => z, 3 / W => w;
    Normal2; 2; 0 / X => x, 1 / Y => y;
    Normal3; 3; 0 / X => x, 1 / Y => y, 2 / Z => z;
);


macro_rules! impl_cross {
    ($($ty:ident),*) => {
        $(
            impl<T: Copy + DifferenceOfProducts> $ty<T> {
                #[inline]
                pub fn cross(self, rhs: Self) -> Self {
                    Self {
                        x: T::difference_of_products(self.y, rhs.z, self.z, rhs.y),
                        y: T::difference_of_products(self.z, rhs.x, self.x, rhs.z),
                        z: T::difference_of_products(self.x, rhs.y, self.y, rhs.x),
                    }
                }
            }
        )*
    }
}

impl_cross!(Point3, Vec3, Normal3);


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

            impl From<$ty<f32>> for $ty<f64> {
                fn from(value: $ty<f32>) -> Self {
                    Self::new($(value.$i as f64,)*)
                }
            }

            impl From<$ty<f64>> for $ty<f32> {
                fn from(value: $ty<f64>) -> Self {
                    Self::new($(value.$i as f32,)*)
                }
            }
        )*
    };
}

impl_interval!(
    Point2 => Float: x,y;
    Point3 => Float: x,y,z;
    Point4 => Float: x,y,z,w;
    Vec2 => Float: x,y;
    Vec3 => Float: x,y,z;
    Vec4 => Float: x,y,z,w;
    Normal2 => Float: x,y;
    Normal3 => Float: x,y,z;
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
    Point2: x,y;
    Point3: x,y,z;
    Point4: x,y,z,w;
    Vec2: x,y;
    Vec3: x,y,z;
    Vec4: x,y,z,w;
    Normal2: x,y;
    Normal3: x,y,z;
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
    Point2: x,y;
    Point3: x,y,z;
    Point4: x,y,z,w;
    Vec2: x,y;
    Vec3: x,y,z;
    Vec4: x,y,z,w;
    Normal2: x,y;
    Normal3: x,y,z;
);


macro_rules! impl_mul_prim {
    ($($prim:ident: $($ty:ident),*);*) => {
        $(
            $(
                impl Mul<$ty<$prim>> for $prim {
                    type Output = $ty<$prim>;
                    #[inline]
                    fn mul(self, rhs: $ty<$prim>) -> Self::Output {
                        rhs * self
                    }
                }
            )*
        )*
    }
}

impl_mul_prim!(f32: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               f64: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               i8: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               i16: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               i32: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               i64: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               i128: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               u8: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               u16: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               u32: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               u64: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               u128: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3;
               Interval: Point2, Point3, Point4, Vec2, Vec3, Vec4, Normal2, Normal3);


impl<T: NumericField + NumericNegative + NumericFloat> Vec3<T> {
    pub fn local_basis(self) -> (Vec3<T>, Vec3<T>) {
        debug_assert!(self.is_normalized());

        let sign = self.z.sign();
        let a = T::NEG_ONE / (sign + self.z);
        let b = self.x * self.y * a;
        (
            Vec3::new(T::ONE + sign * sqr(self.x) * a, sign * b, -sign * self.x),
            Vec3::new(b, sign + sqr(self.y) * a, -self.y)
        )
    }
}

impl<T: PartialOrd> Point3<T> {
    pub fn min_element_index(&self) -> usize {
        if self.x < self.y && self.x < self.z {
            0
        } else if self.y < self.z {
            1
        } else {
            2
        }
    }

    pub fn max_element_index(&self) -> usize {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.z {
            1
        } else {
            2
        }
    }

    pub fn permute(self, perm: (usize, usize, usize)) -> Self
    where
        T: Copy
    {
        Self::new(self[perm.0], self[perm.1], self[perm.2])
    }
}

impl<T: PartialOrd> Vec3<T> {
    pub fn min_element_index(&self) -> usize {
        if self.x < self.y && self.x < self.z {
            0
        } else if self.y < self.z {
            1
        } else {
            2
        }
    }

    pub fn max_element_index(&self) -> usize {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.z {
            1
        } else {
            2
        }
    }

    pub fn permute(self, perm: (usize, usize, usize)) -> Self
    where
        T: Copy
    {
        Self::new(self[perm.0], self[perm.1], self[perm.2])
    }
}

impl<T: PartialOrd> Normal3<T> {
    pub fn min_element_index(&self) -> usize {
        if self.x < self.y && self.x < self.z {
            0
        } else if self.y < self.z {
            1
        } else {
            2
        }
    }

    pub fn max_element_index(&self) -> usize {
        if self.x > self.y && self.x > self.z {
            0
        } else if self.y > self.z {
            1
        } else {
            2
        }
    }

    pub fn permute(self, perm: (usize, usize, usize)) -> Self
    where
        T: Copy
    {
        Self::new(self[perm.0], self[perm.1], self[perm.2])
    }
}

impl Point3<Float> {
    pub fn coordinate_system(&self) -> (Vec3f, Vec3f) {
        let sign = Float::copysign(1.0, self.z);

        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        let v2 = Vec3f::new(1.0 + sign * sqr(self.x) * a, sign * b, -sign * self.x);
        let v3 = Vec3f::new(b, sign + sqr(self.y) * a, -self.y);

        (v2, v3)
    }

    pub fn angle_between(self, v: Self) -> Float {
        debug_assert!(!self.is_nan());
        debug_assert!(!v.is_nan());

        if self.dot(v) < 0.0 {
            PI - 2.0 * safe::asin((self + v).length() / 2.0)
        } else {
            2.0 * safe::asin((v - self).length() / 2.0)
        }
    }
}

impl Vec3<Float> {
    pub fn coordinate_system(&self) -> (Vec3f, Vec3f) {
        let sign = Float::copysign(1.0, self.z);

        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        let v2 = Vec3f::new(1.0 + sign * sqr(self.x) * a, sign * b, -sign * self.x);
        let v3 = Vec3f::new(b, sign + sqr(self.y) * a, -self.y);

        (v2, v3)
    }

    pub fn angle_between(self, v: Self) -> Float {
        debug_assert!(!self.is_nan());
        debug_assert!(!v.is_nan());

        if self.dot(v) < 0.0 {
            PI - 2.0 * safe::asin((self + v).length() / 2.0)
        } else {
            2.0 * safe::asin((v - self).length() / 2.0)
        }
    }
}
