use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};

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
        impl<T> $name<T> {
            #[inline]
            pub fn new($($v: T,)*) -> Self {
                Self {$(
                    $v,
                )*}
            }

            #[inline]
            pub fn splat(v: T) -> Self {
                Self {$(
                    $v: v,
                )*}
            }

            #[inline]
            pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> $name<U> {
                $name::<U> {$(
                    $v: f(self.$v),
                )*}
            }
        }

        impl<T: Numeric + PartialOrd> $name<T> {
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
                $(math_assert!(min.$v < max.$v);)*
                self.max(min).min(max)
            }
        }

        impl<T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>> $name<T> {
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

        impl<T: Numeric + NumericNegative + NumericFloat + PartialOrd + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>> $name<T> {
            #[inline]
            pub fn length_squared(self) -> T {
                self.dot(self)
            }

            #[inline]
            pub fn length(self) -> T {
                self.length_squared().nsqrt()
            }

            #[inline]
            pub fn length_squared_inv(self) -> T {
                self.length_squared().ninv()
            }

            #[inline]
            pub fn length_inv(self) -> T {
                self.length().ninv()
            }

            #[inline]
            pub fn normalize(self) -> Self {
                self * self.length_inv()
            }

            #[inline]
            pub fn is_normalized(self) -> bool {
                (self.length_squared() - T::ONE).nabs() <= T::BIG_EPSILON
            }

            #[inline]
            pub fn is_nan(self) -> bool {
                self.$x.nnan() $(
                    || self.$y.nnan()
                )*
            }

            #[inline]
            pub fn is_finite(self) -> bool {
                self.$x.nfinite() $(
                    && self.$y.nfinite()
                )*
            }

            #[inline]
            pub fn dot(self, rhs: Self) -> T {
                self.$x * rhs.$x $(+ self.$y * rhs.$y)*
            }

            #[inline]
            pub fn project_onto(self, rhs: Self) -> Self {
                rhs * self.dot(rhs) * rhs.length_squared_inv()
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
        }

        impl<T: Default> Default for $name<T> {
            #[inline]
            fn default() -> Self {
                Self {$(
                    $v: T::default(),
                )*}
            }
        }

        impl<T: Mul<T, Output = T>> Mul<T> for $name<T> {
            type Output = $name<T>;
            fn mul(self, rhs: T) -> Self::Output {
                $name {$(
                    $v: self.$v * rhs,
                )*}
            }
        }

        impl<T: Div<T, Output = T> + Numeric + PartialEq> Div<T> for $name<T> {
            type Output = $name<T>;
            fn div(self, rhs: T) -> Self::Output {
                math_assert!(rhs != T::ZERO);
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
                    $(math_assert!(rhs.$z != T::ZERO);)*
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
decl_vect!(TUnitVec2; x,y);
decl_vect!(TUnitVec3; x,y,z);
decl_vect!(TUnitVec4; x,y,z,w);

create_vect!(TPoint2; x, y; TPoint2:TPoint2, TVec2:TPoint2:x-y, TUnitVec2:TPoint2:x-y);
create_vect!(TPoint3; x,y,z; TPoint3:TPoint3, TVec3:TPoint3:x-y-z, TUnitVec3:TPoint3:x-y-z);
create_vect!(TPoint4; x,y,z,w; TPoint4:TPoint4, TVec4:TPoint4:x-y-z-w, TUnitVec4:TPoint4:x-y-z-w);
create_vect!(TVec2; x,y; TVec2:TVec2, TPoint2:TPoint2:x-y, TUnitVec2:TVec2:x-y);
create_vect!(TVec3; x,y,z; TVec3:TVec3, TPoint3:TPoint3:x-y-z, TUnitVec3:TVec3:x-y-z);
create_vect!(TVec4; x,y,z,w; TVec4:TVec4, TPoint4:TPoint4:x-y-z-w, TUnitVec4:TVec4:x-y-z-w);
create_vect!(TUnitVec2; x,y; TUnitVec2:TUnitVec2, TPoint2:TPoint2:x-y, TVec2:TVec2:x-y);
create_vect!(TUnitVec3; x,y,z; TUnitVec3:TUnitVec3, TPoint3:TPoint3:x-y-z, TVec3:TVec3:x-y-z);
create_vect!(TUnitVec4; x,y,z,w; TUnitVec4:TUnitVec4, TPoint4:TPoint4:x-y-z-w, TVec4:TVec4:x-y-z-w);


macro_rules! impl_index {
    ($($ty:ident: $($index:expr => $v:ident),*);* $(;)*) => {
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
        )*
    }
}


impl_index!(
    TPoint2: 0 => x, 1 => y;
    TPoint3: 0 => x, 1 => y, 2 => z;
    TPoint4: 0 => x, 1 => y, 2 => z, 3 => w;
    TVec2: 0 => x, 1 => y;
    TVec3: 0 => x, 1 => y, 2 => z;
    TVec4: 0 => x, 1 => y, 2 => z, 3 => w;
    TUnitVec2: 0 => x, 1 => y;
    TUnitVec3: 0 => x, 1 => y, 2 => z;
    TUnitVec4: 0 => x, 1 => y, 2 => z, 3 => w;
);


macro_rules! impl_cross {
    ($($ty:ident),*) => {
        $(
            impl<T: Sub<T, Output = T> + Mul<T, Output = T>> $ty<T> {
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

impl_cross!(TVec3, TUnitVec3);


macro_rules! impl_new_normalize {
    ($($ty:ident: $($x:ident),*);* $(;)*) => {
        $(
            impl<T: Numeric + NumericNegative + NumericFloat + PartialOrd + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>> $ty<T> {
                #[inline]
                pub fn new_normalize($($x: T,)*) -> Self {
                    Self {$(
                        $x,
                    )*}.normalize()
                }
            }
        )*
    }
}


impl_new_normalize!(TUnitVec2: x,y; TUnitVec3: x,y,z; TUnitVec4: x,y,z,w);


macro_rules! impl_swizzle {
    ($($ty:ident: $($name:ident -> $out:ident[$($i:expr),+]),* $(,)*);* $(;)*) => {
        $(
            impl<T> $ty<T> {
                $(
                    #[inline]
                    pub fn $name(self) -> $out<T> {
                        $out::new($(
                            self[$i],
                        )+)
                    }
                )*
            }
        )*
    }
}


impl_swizzle!(
    TPoint2: 
        xx -> TPoint2[0, 0],
        xy -> TPoint2[0, 1],
        yx -> TPoint2[1, 0],
        yy -> TPoint2[1, 1],
        xxx -> TPoint3[0, 0, 0],
        xxy -> TPoint3[0, 0, 1],
        xyx -> TPoint3[0, 1, 0],
        xyy -> TPoint3[0, 1, 1],
        yxx -> TPoint3[1, 0, 0],
        yxy -> TPoint3[1, 0, 1],
        yyx -> TPoint3[1, 1, 0],
        yyy -> TPoint3[1, 1, 1],
        xxxx -> TPoint4[0, 0, 0, 0],
        xxxy -> TPoint4[0, 0, 0, 1],
        xxyx -> TPoint4[0, 0, 1, 0],
        xxyy -> TPoint4[0, 0, 1, 1],
        xyxx -> TPoint4[0, 1, 0, 0],
        xyxy -> TPoint4[0, 1, 0, 1],
        xyyx -> TPoint4[0, 1, 1, 0],
        xyyy -> TPoint4[0, 1, 1, 1],
        yxxx -> TPoint4[1, 0, 0, 0],
        yxxy -> TPoint4[1, 0, 0, 1],
        yxyx -> TPoint4[1, 0, 1, 0],
        yxyy -> TPoint4[1, 0, 1, 1],
        yyxx -> TPoint4[1, 1, 0, 0],
        yyxy -> TPoint4[1, 1, 0, 1],
        yyyx -> TPoint4[1, 1, 1, 0],
        yyyy -> TPoint4[1, 1, 1, 1];
    TPoint3:
        xx -> TPoint2[0, 0],
        xy -> TPoint2[0, 1],
        xz -> TPoint2[0, 2],
        yx -> TPoint2[1, 0],
        yy -> TPoint2[1, 1],
        yz -> TPoint2[1, 2],
        zx -> TPoint2[2, 0],
        zy -> TPoint2[2, 1],
        zz -> TPoint2[2, 2],
        xxx -> TPoint3[0, 0, 0],
        xxy -> TPoint3[0, 0, 1],
        xxz -> TPoint3[0, 0, 2],
        xyx -> TPoint3[0, 1, 0],
        xyy -> TPoint3[0, 1, 1],
        xyz -> TPoint3[0, 1, 2],
        xzx -> TPoint3[0, 2, 0],
        xzy -> TPoint3[0, 2, 1],
        xzz -> TPoint3[0, 2, 2],
        yxx -> TPoint3[1, 0, 0],
        yxy -> TPoint3[1, 0, 1],
        yxz -> TPoint3[1, 0, 2],
        yyx -> TPoint3[1, 1, 0],
        yyy -> TPoint3[1, 1, 1],
        yyz -> TPoint3[1, 1, 2],
        yzx -> TPoint3[1, 2, 0],
        yzy -> TPoint3[1, 2, 1],
        yzz -> TPoint3[1, 2, 2],
        zxx -> TPoint3[2, 0, 0],
        zxy -> TPoint3[2, 0, 1],
        zxz -> TPoint3[2, 0, 2],
        zyx -> TPoint3[2, 1, 0],
        zyy -> TPoint3[2, 1, 1],
        zyz -> TPoint3[2, 1, 2],
        zzx -> TPoint3[2, 2, 0],
        zzy -> TPoint3[2, 2, 1],
        zzz -> TPoint3[2, 2, 2],
        xxxx -> TPoint4[0, 0, 0, 0],
        xxxy -> TPoint4[0, 0, 0, 1],
        xxxz -> TPoint4[0, 0, 0, 2],
        xxyx -> TPoint4[0, 0, 1, 0],
        xxyy -> TPoint4[0, 0, 1, 1],
        xxyz -> TPoint4[0, 0, 1, 2],
        xxzx -> TPoint4[0, 0, 2, 0],
        xxzy -> TPoint4[0, 0, 2, 1],
        xxzz -> TPoint4[0, 0, 2, 2],
        xyxx -> TPoint4[0, 1, 0, 0],
        xyxy -> TPoint4[0, 1, 0, 1],
        xyxz -> TPoint4[0, 1, 0, 2],
        xyyx -> TPoint4[0, 1, 1, 0],
        xyyy -> TPoint4[0, 1, 1, 1],
        xyyz -> TPoint4[0, 1, 1, 2],
        xyzx -> TPoint4[0, 1, 2, 0],
        xyzy -> TPoint4[0, 1, 2, 1],
        xyzz -> TPoint4[0, 1, 2, 2],
        xzxx -> TPoint4[0, 2, 0, 0],
        xzxy -> TPoint4[0, 2, 0, 1],
        xzxz -> TPoint4[0, 2, 0, 2],
        xzyx -> TPoint4[0, 2, 1, 0],
        xzyy -> TPoint4[0, 2, 1, 1],
        xzyz -> TPoint4[0, 2, 1, 2],
        xzzx -> TPoint4[0, 2, 2, 0],
        xzzy -> TPoint4[0, 2, 2, 1],
        xzzz -> TPoint4[0, 2, 2, 2],
        yxxx -> TPoint4[0, 0, 0, 0],
        yxxy -> TPoint4[0, 0, 0, 1],
        yxxz -> TPoint4[0, 0, 0, 2],
        yxyx -> TPoint4[0, 0, 1, 0],
        yxyy -> TPoint4[0, 0, 1, 1],
        yxyz -> TPoint4[0, 0, 1, 2],
        yxzx -> TPoint4[0, 0, 2, 0],
        yxzy -> TPoint4[0, 0, 2, 1],
        yxzz -> TPoint4[0, 0, 2, 2],
        yyxx -> TPoint4[0, 1, 0, 0],
        yyxy -> TPoint4[0, 1, 0, 1],
        yyxz -> TPoint4[0, 1, 0, 2],
        yyyx -> TPoint4[0, 1, 1, 0],
        yyyy -> TPoint4[0, 1, 1, 1],
        yyyz -> TPoint4[0, 1, 1, 2],
        yyzx -> TPoint4[0, 1, 2, 0],
        yyzy -> TPoint4[0, 1, 2, 1],
        yyzz -> TPoint4[0, 1, 2, 2],
        yzxx -> TPoint4[0, 2, 0, 0],
        yzxy -> TPoint4[0, 2, 0, 1],
        yzxz -> TPoint4[0, 2, 0, 2],
        yzyx -> TPoint4[0, 2, 1, 0],
        yzyy -> TPoint4[0, 2, 1, 1],
        yzyz -> TPoint4[0, 2, 1, 2],
        yzzx -> TPoint4[0, 2, 2, 0],
        yzzy -> TPoint4[0, 2, 2, 1],
        yzzz -> TPoint4[0, 2, 2, 2],
        zxxx -> TPoint4[0, 0, 0, 0],
        zxxy -> TPoint4[0, 0, 0, 1],
        zxxz -> TPoint4[0, 0, 0, 2],
        zxyx -> TPoint4[0, 0, 1, 0],
        zxyy -> TPoint4[0, 0, 1, 1],
        zxyz -> TPoint4[0, 0, 1, 2],
        zxzx -> TPoint4[0, 0, 2, 0],
        zxzy -> TPoint4[0, 0, 2, 1],
        zxzz -> TPoint4[0, 0, 2, 2],
        zyxx -> TPoint4[0, 1, 0, 0],
        zyxy -> TPoint4[0, 1, 0, 1],
        zyxz -> TPoint4[0, 1, 0, 2],
        zyyx -> TPoint4[0, 1, 1, 0],
        zyyy -> TPoint4[0, 1, 1, 1],
        zyyz -> TPoint4[0, 1, 1, 2],
        zyzx -> TPoint4[0, 1, 2, 0],
        zyzy -> TPoint4[0, 1, 2, 1],
        zyzz -> TPoint4[0, 1, 2, 2],
        zzxx -> TPoint4[0, 2, 0, 0],
        zzxy -> TPoint4[0, 2, 0, 1],
        zzxz -> TPoint4[0, 2, 0, 2],
        zzyx -> TPoint4[0, 2, 1, 0],
        zzyy -> TPoint4[0, 2, 1, 1],
        zzyz -> TPoint4[0, 2, 1, 2],
        zzzx -> TPoint4[0, 2, 2, 0],
        zzzy -> TPoint4[0, 2, 2, 1],
        zzzz -> TPoint4[0, 2, 2, 2];
    TPoint4:
        xx -> TPoint2[0, 0],
        xy -> TPoint2[0, 1],
        xz -> TPoint2[0, 2],
        xw -> TPoint2[0, 3],
        yx -> TPoint2[1, 0],
        yy -> TPoint2[1, 1],
        yz -> TPoint2[1, 2],
        yw -> TPoint2[1, 3],
        zx -> TPoint2[2, 0],
        zy -> TPoint2[2, 1],
        zz -> TPoint2[2, 2],
        zw -> TPoint2[2, 3],
        wx -> TPoint2[3, 0],
        wy -> TPoint2[3, 1],
        wz -> TPoint2[3, 2],
        ww -> TPoint2[3, 3],
        xxx -> TPoint3[0, 0, 0],
        xxy -> TPoint3[0, 0, 1],
        xxz -> TPoint3[0, 0, 2],
        xxw -> TPoint3[0, 0, 3],
        xyx -> TPoint3[0, 1, 0],
        xyy -> TPoint3[0, 1, 1],
        xyz -> TPoint3[0, 1, 2],
        xyw -> TPoint3[0, 1, 3],
        xzx -> TPoint3[0, 2, 0],
        xzy -> TPoint3[0, 2, 1],
        xzz -> TPoint3[0, 2, 2],
        xzw -> TPoint3[0, 2, 3],
        xwx -> TPoint3[0, 3, 0],
        xwy -> TPoint3[0, 3, 1],
        xwz -> TPoint3[0, 3, 2],
        xww -> TPoint3[0, 3, 3],
        yxx -> TPoint3[1, 0, 0],
        yxy -> TPoint3[1, 0, 1],
        yxz -> TPoint3[1, 0, 2],
        yxw -> TPoint3[1, 0, 3],
        yyx -> TPoint3[1, 1, 0],
        yyy -> TPoint3[1, 1, 1],
        yyz -> TPoint3[1, 1, 2],
        yyw -> TPoint3[1, 1, 3],
        yzx -> TPoint3[1, 2, 0],
        yzy -> TPoint3[1, 2, 1],
        yzz -> TPoint3[1, 2, 2],
        yzw -> TPoint3[1, 2, 3],
        ywx -> TPoint3[1, 3, 0],
        ywy -> TPoint3[1, 3, 1],
        ywz -> TPoint3[1, 3, 2],
        yww -> TPoint3[1, 3, 3],
        zxx -> TPoint3[2, 0, 0],
        zxy -> TPoint3[2, 0, 1],
        zxz -> TPoint3[2, 0, 2],
        zxw -> TPoint3[2, 0, 3],
        zyx -> TPoint3[2, 1, 0],
        zyy -> TPoint3[2, 1, 1],
        zyz -> TPoint3[2, 1, 2],
        zyw -> TPoint3[2, 1, 3],
        zzx -> TPoint3[2, 2, 0],
        zzy -> TPoint3[2, 2, 1],
        zzz -> TPoint3[2, 2, 2],
        zzw -> TPoint3[2, 2, 3],
        zwx -> TPoint3[2, 3, 0],
        zwy -> TPoint3[2, 3, 1],
        zwz -> TPoint3[2, 3, 2],
        zww -> TPoint3[2, 3, 3],
        wxx -> TPoint3[3, 0, 0],
        wxy -> TPoint3[3, 0, 1],
        wxz -> TPoint3[3, 0, 2],
        wxw -> TPoint3[3, 0, 3],
        wyx -> TPoint3[3, 1, 0],
        wyy -> TPoint3[3, 1, 1],
        wyz -> TPoint3[3, 1, 2],
        wyw -> TPoint3[3, 1, 3],
        wzx -> TPoint3[3, 2, 0],
        wzy -> TPoint3[3, 2, 1],
        wzz -> TPoint3[3, 2, 2],
        wzw -> TPoint3[3, 2, 3],
        wwx -> TPoint3[3, 3, 0],
        wwy -> TPoint3[3, 3, 1],
        wwz -> TPoint3[3, 3, 2],
        www -> TPoint3[3, 3, 3],
        xxxx -> TPoint4[0, 0, 0, 0],
        xxxy -> TPoint4[0, 0, 0, 1],
        xxxz -> TPoint4[0, 0, 0, 2],
        xxxw -> TPoint4[0, 0, 0, 3],
        xxyx -> TPoint4[0, 0, 1, 0],
        xxyy -> TPoint4[0, 0, 1, 1],
        xxyz -> TPoint4[0, 0, 1, 2],
        xxyw -> TPoint4[0, 0, 1, 3],
        xxzx -> TPoint4[0, 0, 2, 0],
        xxzy -> TPoint4[0, 0, 2, 1],
        xxzz -> TPoint4[0, 0, 2, 2],
        xxzw -> TPoint4[0, 0, 2, 3],
        xxwx -> TPoint4[0, 0, 3, 0],
        xxwy -> TPoint4[0, 0, 3, 1],
        xxwz -> TPoint4[0, 0, 3, 2],
        xxww -> TPoint4[0, 0, 3, 3],
        xyxx -> TPoint4[0, 1, 0, 0],
        xyxy -> TPoint4[0, 1, 0, 1],
        xyxz -> TPoint4[0, 1, 0, 2],
        xyxw -> TPoint4[0, 1, 0, 3],
        xyyx -> TPoint4[0, 1, 1, 0],
        xyyy -> TPoint4[0, 1, 1, 1],
        xyyz -> TPoint4[0, 1, 1, 2],
        xyyw -> TPoint4[0, 1, 1, 3],
        xyzx -> TPoint4[0, 1, 2, 0],
        xyzy -> TPoint4[0, 1, 2, 1],
        xyzz -> TPoint4[0, 1, 2, 2],
        xyzw -> TPoint4[0, 1, 2, 3],
        xywx -> TPoint4[0, 1, 3, 0],
        xywy -> TPoint4[0, 1, 3, 1],
        xywz -> TPoint4[0, 1, 3, 2],
        xyww -> TPoint4[0, 1, 3, 3],
        xzxx -> TPoint4[0, 2, 0, 0],
        xzxy -> TPoint4[0, 2, 0, 1],
        xzxz -> TPoint4[0, 2, 0, 2],
        xzxw -> TPoint4[0, 2, 0, 3],
        xzyx -> TPoint4[0, 2, 1, 0],
        xzyy -> TPoint4[0, 2, 1, 1],
        xzyz -> TPoint4[0, 2, 1, 2],
        xzyw -> TPoint4[0, 2, 1, 3],
        xzzx -> TPoint4[0, 2, 2, 0],
        xzzy -> TPoint4[0, 2, 2, 1],
        xzzz -> TPoint4[0, 2, 2, 2],
        xzzw -> TPoint4[0, 2, 2, 3],
        xzwx -> TPoint4[0, 2, 3, 0],
        xzwy -> TPoint4[0, 2, 3, 1],
        xzwz -> TPoint4[0, 2, 3, 2],
        xzww -> TPoint4[0, 2, 3, 3],
        xwxx -> TPoint4[0, 3, 0, 0],
        xwxy -> TPoint4[0, 3, 0, 1],
        xwxz -> TPoint4[0, 3, 0, 2],
        xwxw -> TPoint4[0, 3, 0, 3],
        xwyx -> TPoint4[0, 3, 1, 0],
        xwyy -> TPoint4[0, 3, 1, 1],
        xwyz -> TPoint4[0, 3, 1, 2],
        xwyw -> TPoint4[0, 3, 1, 3],
        xwzx -> TPoint4[0, 3, 2, 0],
        xwzy -> TPoint4[0, 3, 2, 1],
        xwzz -> TPoint4[0, 3, 2, 2],
        xwzw -> TPoint4[0, 3, 2, 3],
        xwwx -> TPoint4[0, 3, 3, 0],
        xwwy -> TPoint4[0, 3, 3, 1],
        xwwz -> TPoint4[0, 3, 3, 2],
        xwww -> TPoint4[0, 3, 3, 3],
        yxxx -> TPoint4[1, 0, 0, 0],
        yxxy -> TPoint4[1, 0, 0, 1],
        yxxz -> TPoint4[1, 0, 0, 2],
        yxxw -> TPoint4[1, 0, 0, 3],
        yxyx -> TPoint4[1, 0, 1, 0],
        yxyy -> TPoint4[1, 0, 1, 1],
        yxyz -> TPoint4[1, 0, 1, 2],
        yxyw -> TPoint4[1, 0, 1, 3],
        yxzx -> TPoint4[1, 0, 2, 0],
        yxzy -> TPoint4[1, 0, 2, 1],
        yxzz -> TPoint4[1, 0, 2, 2],
        yxzw -> TPoint4[1, 0, 2, 3],
        yxwx -> TPoint4[1, 0, 3, 0],
        yxwy -> TPoint4[1, 0, 3, 1],
        yxwz -> TPoint4[1, 0, 3, 2],
        yxww -> TPoint4[1, 0, 3, 3],
        yyxx -> TPoint4[1, 1, 0, 0],
        yyxy -> TPoint4[1, 1, 0, 1],
        yyxz -> TPoint4[1, 1, 0, 2],
        yyxw -> TPoint4[1, 1, 0, 3],
        yyyx -> TPoint4[1, 1, 1, 0],
        yyyy -> TPoint4[1, 1, 1, 1],
        yyyz -> TPoint4[1, 1, 1, 2],
        yyyw -> TPoint4[1, 1, 1, 3],
        yyzx -> TPoint4[1, 1, 2, 0],
        yyzy -> TPoint4[1, 1, 2, 1],
        yyzz -> TPoint4[1, 1, 2, 2],
        yyzw -> TPoint4[1, 1, 2, 3],
        yywx -> TPoint4[1, 1, 3, 0],
        yywy -> TPoint4[1, 1, 3, 1],
        yywz -> TPoint4[1, 1, 3, 2],
        yyww -> TPoint4[1, 1, 3, 3],
        yzxx -> TPoint4[1, 2, 0, 0],
        yzxy -> TPoint4[1, 2, 0, 1],
        yzxz -> TPoint4[1, 2, 0, 2],
        yzxw -> TPoint4[1, 2, 0, 3],
        yzyx -> TPoint4[1, 2, 1, 0],
        yzyy -> TPoint4[1, 2, 1, 1],
        yzyz -> TPoint4[1, 2, 1, 2],
        yzyw -> TPoint4[1, 2, 1, 3],
        yzzx -> TPoint4[1, 2, 2, 0],
        yzzy -> TPoint4[1, 2, 2, 1],
        yzzz -> TPoint4[1, 2, 2, 2],
        yzzw -> TPoint4[1, 2, 2, 3],
        yzwx -> TPoint4[1, 2, 3, 0],
        yzwy -> TPoint4[1, 2, 3, 1],
        yzwz -> TPoint4[1, 2, 3, 2],
        yzww -> TPoint4[1, 2, 3, 3],
        ywxx -> TPoint4[1, 3, 0, 0],
        ywxy -> TPoint4[1, 3, 0, 1],
        ywxz -> TPoint4[1, 3, 0, 2],
        ywxw -> TPoint4[1, 3, 0, 3],
        ywyx -> TPoint4[1, 3, 1, 0],
        ywyy -> TPoint4[1, 3, 1, 1],
        ywyz -> TPoint4[1, 3, 1, 2],
        ywyw -> TPoint4[1, 3, 1, 3],
        ywzx -> TPoint4[1, 3, 2, 0],
        ywzy -> TPoint4[1, 3, 2, 1],
        ywzz -> TPoint4[1, 3, 2, 2],
        ywzw -> TPoint4[1, 3, 2, 3],
        ywwx -> TPoint4[1, 3, 3, 0],
        ywwy -> TPoint4[1, 3, 3, 1],
        ywwz -> TPoint4[1, 3, 3, 2],
        ywww -> TPoint4[1, 3, 3, 3],
        zxxx -> TPoint4[2, 0, 0, 0],
        zxxy -> TPoint4[2, 0, 0, 1],
        zxxz -> TPoint4[2, 0, 0, 2],
        zxxw -> TPoint4[2, 0, 0, 3],
        zxyx -> TPoint4[2, 0, 1, 0],
        zxyy -> TPoint4[2, 0, 1, 1],
        zxyz -> TPoint4[2, 0, 1, 2],
        zxyw -> TPoint4[2, 0, 1, 3],
        zxzx -> TPoint4[2, 0, 2, 0],
        zxzy -> TPoint4[2, 0, 2, 1],
        zxzz -> TPoint4[2, 0, 2, 2],
        zxzw -> TPoint4[2, 0, 2, 3],
        zxwx -> TPoint4[2, 0, 3, 0],
        zxwy -> TPoint4[2, 0, 3, 1],
        zxwz -> TPoint4[2, 0, 3, 2],
        zxww -> TPoint4[2, 0, 3, 3],
        zyxx -> TPoint4[2, 1, 0, 0],
        zyxy -> TPoint4[2, 1, 0, 1],
        zyxz -> TPoint4[2, 1, 0, 2],
        zyxw -> TPoint4[2, 1, 0, 3],
        zyyx -> TPoint4[2, 1, 1, 0],
        zyyy -> TPoint4[2, 1, 1, 1],
        zyyz -> TPoint4[2, 1, 1, 2],
        zyyw -> TPoint4[2, 1, 1, 3],
        zyzx -> TPoint4[2, 1, 2, 0],
        zyzy -> TPoint4[2, 1, 2, 1],
        zyzz -> TPoint4[2, 1, 2, 2],
        zyzw -> TPoint4[2, 1, 2, 3],
        zywx -> TPoint4[2, 1, 3, 0],
        zywy -> TPoint4[2, 1, 3, 1],
        zywz -> TPoint4[2, 1, 3, 2],
        zyww -> TPoint4[2, 1, 3, 3],
        zzxx -> TPoint4[2, 2, 0, 0],
        zzxy -> TPoint4[2, 2, 0, 1],
        zzxz -> TPoint4[2, 2, 0, 2],
        zzxw -> TPoint4[2, 2, 0, 3],
        zzyx -> TPoint4[2, 2, 1, 0],
        zzyy -> TPoint4[2, 2, 1, 1],
        zzyz -> TPoint4[2, 2, 1, 2],
        zzyw -> TPoint4[2, 2, 1, 3],
        zzzx -> TPoint4[2, 2, 2, 0],
        zzzy -> TPoint4[2, 2, 2, 1],
        zzzz -> TPoint4[2, 2, 2, 2],
        zzzw -> TPoint4[2, 2, 2, 3],
        zzwx -> TPoint4[2, 2, 3, 0],
        zzwy -> TPoint4[2, 2, 3, 1],
        zzwz -> TPoint4[2, 2, 3, 2],
        zzww -> TPoint4[2, 2, 3, 3],
        zwxx -> TPoint4[2, 3, 0, 0],
        zwxy -> TPoint4[2, 3, 0, 1],
        zwxz -> TPoint4[2, 3, 0, 2],
        zwxw -> TPoint4[2, 3, 0, 3],
        zwyx -> TPoint4[2, 3, 1, 0],
        zwyy -> TPoint4[2, 3, 1, 1],
        zwyz -> TPoint4[2, 3, 1, 2],
        zwyw -> TPoint4[2, 3, 1, 3],
        zwzx -> TPoint4[2, 3, 2, 0],
        zwzy -> TPoint4[2, 3, 2, 1],
        zwzz -> TPoint4[2, 3, 2, 2],
        zwzw -> TPoint4[2, 3, 2, 3],
        zwwx -> TPoint4[2, 3, 3, 0],
        zwwy -> TPoint4[2, 3, 3, 1],
        zwwz -> TPoint4[2, 3, 3, 2],
        zwww -> TPoint4[2, 3, 3, 3],
        wxxx -> TPoint4[3, 0, 0, 0],
        wxxy -> TPoint4[3, 0, 0, 1],
        wxxz -> TPoint4[3, 0, 0, 2],
        wxxw -> TPoint4[3, 0, 0, 3],
        wxyx -> TPoint4[3, 0, 1, 0],
        wxyy -> TPoint4[3, 0, 1, 1],
        wxyz -> TPoint4[3, 0, 1, 2],
        wxyw -> TPoint4[3, 0, 1, 3],
        wxzx -> TPoint4[3, 0, 2, 0],
        wxzy -> TPoint4[3, 0, 2, 1],
        wxzz -> TPoint4[3, 0, 2, 2],
        wxzw -> TPoint4[3, 0, 2, 3],
        wxwx -> TPoint4[3, 0, 3, 0],
        wxwy -> TPoint4[3, 0, 3, 1],
        wxwz -> TPoint4[3, 0, 3, 2],
        wxww -> TPoint4[3, 0, 3, 3],
        wyxx -> TPoint4[3, 1, 0, 0],
        wyxy -> TPoint4[3, 1, 0, 1],
        wyxz -> TPoint4[3, 1, 0, 2],
        wyxw -> TPoint4[3, 1, 0, 3],
        wyyx -> TPoint4[3, 1, 1, 0],
        wyyy -> TPoint4[3, 1, 1, 1],
        wyyz -> TPoint4[3, 1, 1, 2],
        wyyw -> TPoint4[3, 1, 1, 3],
        wyzx -> TPoint4[3, 1, 2, 0],
        wyzy -> TPoint4[3, 1, 2, 1],
        wyzz -> TPoint4[3, 1, 2, 2],
        wyzw -> TPoint4[3, 1, 2, 3],
        wywx -> TPoint4[3, 1, 3, 0],
        wywy -> TPoint4[3, 1, 3, 1],
        wywz -> TPoint4[3, 1, 3, 2],
        wyww -> TPoint4[3, 1, 3, 3],
        wzxx -> TPoint4[3, 2, 0, 0],
        wzxy -> TPoint4[3, 2, 0, 1],
        wzxz -> TPoint4[3, 2, 0, 2],
        wzxw -> TPoint4[3, 2, 0, 3],
        wzyx -> TPoint4[3, 2, 1, 0],
        wzyy -> TPoint4[3, 2, 1, 1],
        wzyz -> TPoint4[3, 2, 1, 2],
        wzyw -> TPoint4[3, 2, 1, 3],
        wzzx -> TPoint4[3, 2, 2, 0],
        wzzy -> TPoint4[3, 2, 2, 1],
        wzzz -> TPoint4[3, 2, 2, 2],
        wzzw -> TPoint4[3, 2, 2, 3],
        wzwx -> TPoint4[3, 2, 3, 0],
        wzwy -> TPoint4[3, 2, 3, 1],
        wzwz -> TPoint4[3, 2, 3, 2],
        wzww -> TPoint4[3, 2, 3, 3],
        wwxx -> TPoint4[3, 3, 0, 0],
        wwxy -> TPoint4[3, 3, 0, 1],
        wwxz -> TPoint4[3, 3, 0, 2],
        wwxw -> TPoint4[3, 3, 0, 3],
        wwyx -> TPoint4[3, 3, 1, 0],
        wwyy -> TPoint4[3, 3, 1, 1],
        wwyz -> TPoint4[3, 3, 1, 2],
        wwyw -> TPoint4[3, 3, 1, 3],
        wwzx -> TPoint4[3, 3, 2, 0],
        wwzy -> TPoint4[3, 3, 2, 1],
        wwzz -> TPoint4[3, 3, 2, 2],
        wwzw -> TPoint4[3, 3, 2, 3],
        wwwx -> TPoint4[3, 3, 3, 0],
        wwwy -> TPoint4[3, 3, 3, 1],
        wwwz -> TPoint4[3, 3, 3, 2],
        wwww -> TPoint4[3, 3, 3, 3],
);
