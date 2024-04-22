use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use crate::math::*;

macro_rules! create_mat {
    ($name:ident; $vect:ident; $row:expr, $col:expr; $($i:expr;$($j:expr;$m:ident)-*),* $(;)*) => {
        #[derive(Clone, Copy, PartialEq)]
        pub struct $name<T>([[T; $col]; $row]);
        
        impl<T> $name<T> {
            #[inline]
            pub const fn new($($($m: T,)*)*) -> Self {
                Self([$([$($m,)*],)*])
            }
        }

        impl<T: Clone + Copy> $name<T> {
            pub const fn rowc<const N: usize>(self) -> $vect<T> {
                $vect::<T>::new($(self.0[N][$i],)*)
            }

            pub const fn colc<const N: usize>(self) -> $vect<T> {
                $vect::<T>::new($(self.0[$i][N],)*)
            }

            pub const fn index<const R: usize, const C: usize>(self) -> T {
                self.0[R][C]
            }

            #[inline]
            pub fn row(self, n: usize) -> $vect<T> {
                $vect::<T>::from(self.0[n])
            }

            #[inline]
            pub fn col(self, n: usize) -> $vect<T> {
                $vect::<T>::new($(self[($i, n)],)*)
            }
        }

        impl<T> Index<(usize, usize)> for $name<T> {
            type Output = T;

            #[doc = "Indexes into the matrix via `(row, col)`."]
            #[inline]
            fn index(&self, index: (usize, usize)) -> &Self::Output {
                &self.0[index.0][index.1]
            }
        }

        impl<T> IndexMut<(usize, usize)> for $name<T> {
            #[doc = "Mutably indexes into the matrix via `(row, col)`."]
            #[inline]
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                &mut self.0[index.0][index.1]
            }
        }

        impl<T: Clone + Copy + Add<T, Output = T> + Mul<T, Output = T>> Mul<$name<T>> for $name<T> {
            type Output = $name<T>;

            fn mul(self, rhs: Self) -> Self::Output {
                Self([$(
                    [$(self.rowc::<$i>().dot(rhs.colc::<$j>()),)*],
                )*])
            }
        }

        impl<T: Clone + Copy + Mul<T, Output = T>> Mul<T> for $name<T> {
            type Output = $name<T>;

            fn mul(self, rhs: T) -> Self::Output {
                Self([$(
                    [$(self[($i, $j)] * rhs,)*],
                )*])
            }
        }

        impl<T: std::fmt::Debug> std::fmt::Debug for $name<T> {
            #[inline]
            fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(fmt)
            }
        }
    }
}


create_mat!(TMat2; TVec2; 2, 2; 0;0;m00-1;m01,1;0;m10-1;m11);
create_mat!(TMat3; TVec3; 3, 3; 0;0;m00-1;m01-2;m02,1;0;m10-1;m11-2;m12,2;0;m20-1;m21-2;m22);
create_mat!(TMat4; TVec4; 4, 4; 0;0;m00-1;m01-2;m02-3;m03,1;0;m10-1;m11-2;m12-3;m13,2;0;m20-1;m21-2;m22-3;m23,3;0;m30-1;m31-2;m32-3;m33);


macro_rules! impl_mul_vect {
    ($name:ident -> $($vect:ident: $($i:expr),*);*) => {
        $(
            impl<T: Clone + Copy + Add<T, Output = T> + Mul<T, Output = T>> Mul<$vect<T>> for $name<T> {
                type Output = $vect<T>;

                fn mul(self, rhs: $vect<T>) -> $vect<T> {
                    $vect::<T>::new($(
                        $vect::<T>::from(self.rowc::<$i>()).dot(rhs),
                    )*)
                }
            }
        )*
    }
}

impl_mul_vect!(TMat2 -> TVec2:0,1; TPoint2:0,1; TNormal2:0,1);
impl_mul_vect!(TMat3 -> TVec3:0,1,2; TPoint3:0,1,2; TNormal3:0,1,2);
impl_mul_vect!(TMat4 -> TVec4:0,1,2,3; TPoint4:0,1,2,3);


macro_rules! impl_interval {
    ($($ty:ident -> $prim:ident: $($i:tt,$j:tt)-*);*) => {
        $(
            impl<T: NumericField + FloatInterval> From<$ty<$prim>> for $ty<T> {
                fn from(value: $ty<$prim>) -> Self {
                    Self::new($(
                        T::new_interval(value.index::<$i, $j>(), value.index::<$i, $j>()),
                    )*)
                }
            }
        )*
    }
}

impl_interval!(TMat2 -> Scalar: 0,0-0,1-1,0-1,1);
impl_interval!(TMat3 -> Scalar: 0,0-0,1-0,2-1,0-1,1-1,2-2,0-2,1-2,2);
impl_interval!(TMat4 -> Scalar: 0,0-0,1-0,2-0,3-1,0-1,1-1,2-1,3-2,0-2,1-2,2-2,3-3,0-3,1-3,2-3,3);


impl<T: Numeric> TMat2<T> {
    pub const IDENTITY: Self = Self::new(
        T::ONE,  T::ZERO,
        T::ZERO, T::ONE,
    );
}

impl<T: Clone + Copy> TMat2<T> {
    #[inline]
    pub fn transpose(self) -> Self {
        Self::new(
            self[(0, 0)], self[(1, 0)],
            self[(0, 1)], self[(1, 1)],
        )
    }
}

impl<T> TMat2<T> 
where 
    T: Clone + Copy + Numeric + NumericFloat + PartialEq + Neg<Output = T> + Mul<T, Output = T> + Sub<T, Output = T>
{
    #[inline]
    pub fn determinant(self) -> T {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let inv_det = {
            let det = self.determinant();
            math_assert!(det != T::ZERO);
            det.ninv()
        };
        Self::new(
            self.rowc::<1>().y * inv_det,
            self.rowc::<0>().y * -inv_det,
            self.rowc::<1>().x * -inv_det,
            self.rowc::<0>().x * inv_det,
        )
    }
}


impl<T: Numeric> TMat3<T> {
    pub const IDENTITY: Self = Self::new(
        T::ONE,  T::ZERO, T::ZERO,
        T::ZERO, T::ONE,  T::ZERO,
        T::ZERO, T::ZERO, T::ONE,
    );
}

impl<T: Clone + Copy> TMat3<T> {
    #[inline]
    pub fn transpose(self) -> Self {
        Self::new(
            self[(0, 0)], self[(1, 0)], self[(2, 0)],
            self[(0, 1)], self[(1, 1)], self[(2, 1)],
            self[(0, 2)], self[(1, 2)], self[(2, 2)],
        )
    }
}

impl<T> TMat3<T> 
where 
    T: Clone + Copy + Numeric + NumericFloat + PartialEq + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>
{
    #[inline]
    pub fn determinant(self) -> T {
        self.rowc::<2>().dot(self.rowc::<0>().cross(self.rowc::<1>()))
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let tmp0 = self.rowc::<1>().cross(self.rowc::<2>());
        let tmp1 = self.rowc::<2>().cross(self.rowc::<0>());
        let tmp2 = self.rowc::<0>().cross(self.rowc::<1>());
        let det = self.rowc::<2>().dot(tmp2);
        math_assert!(det != T::ZERO);
        let inv_det = TVec3::splat(det.ninv());
        Self([tmp0.mul(inv_det).into(), tmp1.mul(inv_det).into(), tmp2.mul(inv_det).into()]).transpose()
    }
}


impl<T: Numeric> TMat4<T> {
    pub const IDENTITY: Self = Self::new(
        T::ONE,  T::ZERO, T::ZERO, T::ZERO,
        T::ZERO, T::ONE,  T::ZERO, T::ZERO,
        T::ZERO, T::ZERO, T::ONE,  T::ZERO,
        T::ZERO, T::ZERO, T::ZERO, T::ONE,
    );
}

impl<T: Clone + Copy> TMat4<T> {
    #[inline]
    pub fn transpose(self) -> Self {
        Self::new(
            self[(0, 0)], self[(1, 0)], self[(2, 0)], self[(3, 0)],
            self[(0, 1)], self[(1, 1)], self[(2, 1)], self[(3, 1)],
            self[(0, 2)], self[(1, 2)], self[(2, 2)], self[(3, 2)],
            self[(0, 3)], self[(1, 3)], self[(2, 3)], self[(3, 3)],
        )
    }
}

impl<T> TMat4<T>
where 
    T: Clone + Copy + Numeric + NumericNegative + NumericFloat + PartialEq + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>
{
    #[inline]
    pub fn determinant(self) -> T {
        let a2323 = self[(2, 2)] * self[(3, 3)] - self[(2, 3)] * self[(3, 2)];
        let a1323 = self[(2, 1)] * self[(3, 3)] - self[(2, 3)] * self[(3, 1)];
        let a1223 = self[(2, 1)] * self[(3, 2)] - self[(2, 2)] * self[(3, 1)];
        let a0323 = self[(2, 0)] * self[(3, 3)] - self[(2, 3)] * self[(3, 0)];
        let a0223 = self[(2, 0)] * self[(3, 2)] - self[(2, 2)] * self[(3, 0)];
        let a0123 = self[(2, 0)] * self[(3, 1)] - self[(2, 1)] * self[(3, 0)];

        self[(0, 0)] * (self[(1, 1)] * a2323 - self[(1, 2)] * a1323 + self[(1, 3)] * a1223)
            - self[(0, 1)] * (self[(1, 0)] * a2323 - self[(1, 2)] * a0323 + self[(1, 3)] * a0223)
            + self[(0, 2)] * (self[(1, 0)] * a1323 - self[(1, 1)] * a0323 + self[(1, 3)] * a0123)
            - self[(0, 3)] * (self[(1, 0)] * a1223 - self[(1, 1)] * a0223 + self[(1, 2)] * a0123)
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let (m00, m01, m02, m03) = self.rowc::<0>().into();
        let (m10, m11, m12, m13) = self.rowc::<1>().into();
        let (m20, m21, m22, m23) = self.rowc::<2>().into();
        let (m30, m31, m32, m33) = self.rowc::<3>().into();

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;

        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;

        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;

        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;

        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;

        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = TVec4::new(coef00, coef00, coef02, coef03);
        let fac1 = TVec4::new(coef04, coef04, coef06, coef07);
        let fac2 = TVec4::new(coef08, coef08, coef10, coef11);
        let fac3 = TVec4::new(coef12, coef12, coef14, coef15);
        let fac4 = TVec4::new(coef16, coef16, coef18, coef19);
        let fac5 = TVec4::new(coef20, coef20, coef22, coef23);

        let vec0 = TVec4::new(m10, m00, m00, m00);
        let vec1 = TVec4::new(m11, m01, m01, m01);
        let vec2 = TVec4::new(m12, m02, m02, m02);
        let vec3 = TVec4::new(m13, m03, m03, m03);

        let inv0 = vec1.mul(fac0).sub(vec2.mul(fac1)).add(vec3.mul(fac2));
        let inv1 = vec0.mul(fac0).sub(vec2.mul(fac3)).add(vec3.mul(fac4));
        let inv2 = vec0.mul(fac1).sub(vec1.mul(fac3)).add(vec3.mul(fac5));
        let inv3 = vec0.mul(fac2).sub(vec1.mul(fac4)).add(vec2.mul(fac5));

        let sign_a = TVec4::new(T::ONE, T::NEG_ONE, T::ONE, T::NEG_ONE);
        let sign_b = TVec4::new(T::NEG_ONE, T::ONE, T::NEG_ONE, T::ONE);

        let inverse = Self([
            inv0.mul(sign_a).into(),
            inv1.mul(sign_b).into(),
            inv2.mul(sign_a).into(),
            inv3.mul(sign_b).into(),
        ]);

        let col0 = TVec4::new(
            inverse.rowc::<0>().x,
            inverse.rowc::<1>().x,
            inverse.rowc::<2>().x,
            inverse.rowc::<3>().x,
        );

        let dot0 = self.rowc::<0>().mul(col0);
        let dot1 = dot0.x + dot0.y + dot0.z + dot0.w;

        math_assert!(dot1 != T::ZERO);

        let rcp_det = dot1.ninv();
        inverse.mul(rcp_det)
    }
}


impl<T> TMat4<T>
where 
    T: Clone + Copy + Numeric + Neg<Output = T> + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T>
{
    pub fn from_translation(delta: TPoint3<T>) -> Self {
        Self::new(
            T::ONE,  T::ZERO, T::ZERO, delta.x,
            T::ZERO, T::ONE,  T::ZERO, delta.y,
            T::ZERO, T::ZERO, T::ONE,  delta.z,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,
        )
    }

    pub fn from_rotation_x(sin_theta: T, cos_theta: T) -> Self {
        Self::new(
            T::ONE, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, cos_theta, -sin_theta, T::ZERO,
            T::ZERO, sin_theta, cos_theta, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE
        )
    }

    pub fn from_rotation_y(sin_theta: T, cos_theta: T) -> Self {
        Self::new(
            cos_theta, T::ZERO, sin_theta, T::ZERO,
            T::ZERO, T::ONE, T::ZERO, T::ZERO,
            -sin_theta, T::ZERO, cos_theta, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE
        )
    }

    pub fn from_rotation_z(sin_theta: T, cos_theta: T) -> Self {
        Self::new(
            cos_theta, -sin_theta, T::ZERO, T::ZERO,
            sin_theta, cos_theta, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, T::ONE, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE
        )
    }

    pub fn from_rotation(sin_theta: T, cos_theta: T, axis: TVec3<T>) -> Self {
        Self::new(
            axis.x * axis.x + (T::ONE - axis.x * axis.x) * cos_theta,
            axis.x * axis.y * (T::ONE - cos_theta) - axis.z * sin_theta,
            axis.x * axis.z * (T::ONE - cos_theta) + axis.y * sin_theta,
            T::ZERO,
            axis.x * axis.y * (T::ONE - cos_theta) + axis.z * sin_theta,
            axis.y * axis.y + (T::ONE - axis.y * axis.y) * cos_theta,
            axis.y * axis.z * (T::ONE - cos_theta) - axis.x * sin_theta,
            T::ZERO,
            axis.x * axis.z * (T::ONE - cos_theta) - axis.y * sin_theta,
            axis.y * axis.z * (T::ONE - cos_theta) + axis.x * sin_theta,
            axis.z * axis.z + (T::ONE - axis.z * axis.z) * cos_theta,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            T::ONE,
        )
    }

    pub fn from_scale(s: TVec3<T>) -> Self {
        Self::new(
            s.x, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, s.y, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, s.z, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,
        )
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use glam::Vec4Swizzles;

    use crate::math::*;

    #[test]
    fn translate() {
        let m = Mat4::from_translation(Point3f::new(1.0, 2.0, 3.0));
        let p = Point4f::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!((m * p).xyz(), Point3f::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn rotate_axis() {
        let theta = 1.214f32;
        let m = Mat4::from_rotation_x(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_x(theta);

        let p = Point4f::new(0.0, 1.0, 0.0, 1.0);
        let gp = glam::Vec4::new(0.0, 1.0, 0.0, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()));

        let theta = 2.3542f32;
        let m = Mat4::from_rotation_y(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_y(theta);

        let p = Point4f::new(0.3, 7.1, 3.4, 1.0);
        let gp = glam::Vec4::new(0.3, 7.1, 3.4, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()));

        let theta = 0.761f32;
        let m = Mat4::from_rotation_x(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_x(theta);

        let p = Point4f::new(0.146, 3.0135, 7.0541, 1.0);
        let gp = glam::Vec4::new(0.146, 3.0135, 7.0541, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()));
    }

    #[test]
    fn rotate() {
        let theta = 1.431f32;

        let m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(3.0, 1.0, 2.0).normalize());
        let gm = glam::Mat4::from_axis_angle(glam::Vec3::new(3.0, 1.0, 2.0).normalize(), theta);

        let p = Point4f::new(0.413, 4.21, 3.73, 1.0);
        let gp = glam::Vec4::new(0.413, 4.21, 3.73, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()))
    }

    #[test]
    fn scale() {
        let m = Mat4::from_scale(Vec3f::new(1.541, 3.312, 0.231));
        let gm = glam::Mat4::from_scale(glam::Vec3::new(1.541, 3.312, 0.231));

        let p = Point4f::new(1.143, 0.513, 4.041, 1.0);
        let gp = glam::Vec4::new(1.143, 0.513, 4.041, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()))
    }

    #[test]
    fn concat() {
        let theta = 0.712f32;
        let mut m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(4.0, 2.0, 5.0).normalize());
        let mut gm = glam::Mat4::from_axis_angle(glam::Vec3::new(4.0, 2.0, 5.0).normalize(), theta);

        m = m * Mat4::from_translation(Point3f::new(3.02, 2.43, 0.42));
        gm = gm.mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(3.02, 2.43, 0.42)));

        let theta = -1.245f32;
        m = m * Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(-2.0, -3.0, 4.0).normalize());
        gm = gm.mul_mat4(&glam::Mat4::from_axis_angle(glam::Vec3::new(-2.0, -3.0, 4.0).normalize(), theta));

        m = m * Mat4::from_scale(Vec3f::new(1.43, 0.53, 2.34));
        gm = gm.mul_mat4(&glam::Mat4::from_scale(glam::Vec3::new(1.43, 0.53, 2.34)));

        let p = Point4f::new(0.51, 2.12, 4.53, 1.0);
        let gp = glam::Vec4::new(0.51, 2.12, 4.53, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()), epsilon = 2e-6f32);
    }

    #[test]
    fn determinant() {
        let m = Mat4::from_scale(Vec3f::new(2.0, 3.0, 1.5));
        assert_abs_diff_eq!(m.determinant(), 9.0);
    }

    #[test]
    fn inverse() {
        let theta = -0.834f32;
        let m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(2.0, 5.0, 7.0).normalize())
            * Mat4::from_translation(Point3f::new(0.9, 0.3, 0.6))
            * Mat4::from_scale(Vec3f::new(0.8, 1.2, 2.3));
        let gm = glam::Mat4::from_axis_angle(glam::Vec3::new(2.0, 5.0, 7.0).normalize(), theta)
            * glam::Mat4::from_translation(glam::Vec3::new(0.9, 0.3, 0.6))
            * glam::Mat4::from_scale(glam::Vec3::new(0.8, 1.2, 2.3));
        
        let p = Point4f::new(2.23, 0.83, 1.74, 1.0);
        let gp = glam::Vec4::new(2.23, 0.83, 1.74, 1.0);

        let p1 = (m.inverse() * p).xyz();
        let gp1 = (gm.inverse() * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::from(gp1.to_array()));
    }
}
