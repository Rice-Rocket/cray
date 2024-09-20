use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
use compensated_float::CompensatedFloat;
use numeric::DifferenceOfProducts;

use crate::math::*;

macro_rules! create_mat {
    ($name:ident; $vect:ident; $row:expr, $col:expr; $($i:expr;$($j:expr;$m:ident)-*),* $(;)*) => {
        #[derive(Clone, Copy, PartialEq)]
        pub struct $name<T>([[T; $col]; $row]);
        
        impl<T> $name<T> {
            #[inline]
            #[allow(clippy::too_many_arguments)]
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

        impl<T: ToString> std::fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "\n")?;
                let mut size = [0; $col];
                for col in 0..$col {
                    for row in 0..$row {
                        size[col] = size[col].max(self[(row, col)].to_string().len());
                    }
                }
                
                for row in 0..$row {
                    write!(f, "[")?;

                    for col in 0..$col {
                        let spacing = size[col] - self[(row, col)].to_string().len() + 1;
                        write!(f, "{}", " ".repeat(spacing))?;
                        write!(f, "{}", self[(row, col)].to_string())?;
                    }

                    write!(f, " ]")?;
                    if row != $row - 1 {
                        write!(f, "\n")?;
                    }
                }
                Ok(())
            }
        }
    }
}

create_mat!(TMat2; Vec2; 2, 2; 0;0;m00-1;m01,1;0;m10-1;m11);
create_mat!(TMat3; Vec3; 3, 3; 0;0;m00-1;m01-2;m02,1;0;m10-1;m11-2;m12,2;0;m20-1;m21-2;m22);
create_mat!(TMat4; Vec4; 4, 4; 0;0;m00-1;m01-2;m02-3;m03,1;0;m10-1;m11-2;m12-3;m13,2;0;m20-1;m21-2;m22-3;m23,3;0;m30-1;m31-2;m32-3;m33);

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

impl_mul_vect!(TMat2 -> Vec2:0,1; Point2:0,1; Normal2:0,1);
impl_mul_vect!(TMat3 -> Vec3:0,1,2; Point3:0,1,2; Normal3:0,1,2);
impl_mul_vect!(TMat4 -> Vec4:0,1,2,3; Point4:0,1,2,3);

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

impl_interval!(TMat2 -> Float: 0,0-0,1-1,0-1,1);
impl_interval!(TMat3 -> Float: 0,0-0,1-0,2-1,0-1,1-1,2-2,0-2,1-2,2);
impl_interval!(TMat4 -> Float: 0,0-0,1-0,2-0,3-1,0-1,1-1,2-1,3-2,0-2,1-2,2-2,3-3,0-3,1-3,2-3,3);

macro_rules! impl_default {
    ($($ty:ident),*) => {
        $(
            impl<T: NumericConsts> Default for $ty<T> {
                fn default() -> Self {
                    Self::IDENTITY
                }
            }
        )*
    }
}

impl_default!(TMat2, TMat3, TMat4);

impl<T: Copy> From<[T; 4]> for TMat2<T> {
    fn from(value: [T; 4]) -> Self {
        Self::new(value[0], value[1], value[2], value[3])
    }
}

impl<T: Copy> From<[T; 9]> for TMat3<T> {
    fn from(value: [T; 9]) -> Self {
        Self::new(
            value[0], value[1], value[2],
            value[3], value[4], value[5],
            value[6], value[7], value[8],
        )
    }
}

impl<T: Copy> From<[T; 16]> for TMat4<T> {
    fn from(value: [T; 16]) -> Self {
        Self::new(
            value[0], value[1], value[2], value[3],
            value[4], value[5], value[6], value[7],
            value[8], value[9], value[10], value[11],
            value[12], value[13], value[14], value[15],
        )
    }
}

impl<T: Copy> From<[[T; 4]; 4]> for TMat4<T> {
    fn from(value: [[T; 4]; 4]) -> Self {
        Self::new(
            value[0][0], value[0][1], value[0][2], value[0][3],
            value[1][0], value[1][1], value[1][2], value[1][3],
            value[2][0], value[2][1], value[2][2], value[2][3],
            value[3][0], value[3][1], value[3][2], value[3][3],
        )
    }
}

impl<T: NumericConsts> TMat2<T> {
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
    T: Clone + Copy + NumericConsts + NumericFloat + PartialEq 
        + Neg<Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + DifferenceOfProducts
{
    #[inline]
    pub fn determinant(self) -> T {
        T::difference_of_products(self[(0, 0)], self[(1, 1)], self[(0, 1)], self[(1, 0)])
    }

    #[inline]
    pub fn inverse(self) -> Self {
        self.try_inverse().unwrap()
    }

    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let inv_det = {
            let det = self.determinant();
            
            if det == T::ZERO {
                return None;
            }

            det.ninv()
        };

        Some(Self::new(
            self.rowc::<1>().y * inv_det,
            self.rowc::<0>().y * -inv_det,
            self.rowc::<1>().x * -inv_det,
            self.rowc::<0>().x * inv_det,
        ))
    }
}


impl<T: NumericConsts> TMat3<T> {
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

impl TMat3<Float> {
    #[inline]
    pub fn determinant(self) -> Float {
        let minor12 = Float::difference_of_products(self[(1, 1)], self[(2, 2)], self[(1, 2)], self[(2, 1)]);
        let minor02 = Float::difference_of_products(self[(1, 0)], self[(2, 2)], self[(1, 2)], self[(2, 0)]);
        let minor01 = Float::difference_of_products(self[(1, 0)], self[(2, 1)], self[(1, 1)], self[(2, 0)]);
        Float::mul_add(
            self[(0, 2)],
            minor01,
            Float::difference_of_products(self[(0, 0)], minor12, self[(0, 1)], minor02),
        )
    }

    #[inline]
    pub fn inverse(self) -> Self {
        self.try_inverse().unwrap()
    }

    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }

        let inv_det = 1.0 / det;

        let mut out = Mat3::IDENTITY;

        out[(0, 0)] =
            inv_det * Float::difference_of_products(self[(1, 1)], self[(2, 2)], self[(1, 2)], self[(2, 1)]);
        out[(1, 0)] =
            inv_det * Float::difference_of_products(self[(1, 2)], self[(2, 0)], self[(1, 0)], self[(2, 2)]);
        out[(2, 0)] =
            inv_det * Float::difference_of_products(self[(1, 0)], self[(2, 1)], self[(1, 1)], self[(2, 0)]);
        out[(0, 1)] =
            inv_det * Float::difference_of_products(self[(0, 2)], self[(2, 1)], self[(0, 1)], self[(2, 2)]);
        out[(1, 1)] =
            inv_det * Float::difference_of_products(self[(0, 0)], self[(2, 2)], self[(0, 2)], self[(2, 0)]);
        out[(2, 1)] =
            inv_det * Float::difference_of_products(self[(0, 1)], self[(2, 0)], self[(0, 0)], self[(2, 1)]);
        out[(0, 2)] =
            inv_det * Float::difference_of_products(self[(0, 1)], self[(1, 2)], self[(0, 2)], self[(1, 1)]);
        out[(1, 2)] =
            inv_det * Float::difference_of_products(self[(0, 2)], self[(1, 0)], self[(0, 0)], self[(1, 2)]);
        out[(2, 2)] =
            inv_det * Float::difference_of_products(self[(0, 0)], self[(1, 1)], self[(0, 1)], self[(1, 0)]);

        Some(out)
    }
}


impl<T: NumericConsts> TMat4<T> {
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

impl TMat4<Float> {
    #[inline]
    pub fn determinant(self) -> Float {
        let s0 = Float::difference_of_products(self[(0, 0)], self[(1, 1)], self[(1, 0)], self[(0, 1)]);
        let s1 = Float::difference_of_products(self[(0, 0)], self[(1, 2)], self[(1, 0)], self[(0, 2)]);
        let s2 = Float::difference_of_products(self[(0, 0)], self[(1, 3)], self[(1, 0)], self[(0, 3)]);

        let s3 = Float::difference_of_products(self[(0, 1)], self[(1, 2)], self[(1, 1)], self[(0, 2)]);
        let s4 = Float::difference_of_products(self[(0, 1)], self[(1, 3)], self[(1, 1)], self[(0, 3)]);
        let s5 = Float::difference_of_products(self[(0, 2)], self[(1, 3)], self[(1, 2)], self[(0, 3)]);

        let c0 = Float::difference_of_products(self[(2, 0)], self[(3, 1)], self[(3, 0)], self[(2, 1)]);
        let c1 = Float::difference_of_products(self[(2, 0)], self[(3, 2)], self[(3, 0)], self[(2, 2)]);
        let c2 = Float::difference_of_products(self[(2, 0)], self[(3, 3)], self[(3, 0)], self[(2, 3)]);

        let c3 = Float::difference_of_products(self[(2, 1)], self[(3, 2)], self[(3, 1)], self[(2, 2)]);
        let c4 = Float::difference_of_products(self[(2, 1)], self[(3, 3)], self[(3, 1)], self[(2, 3)]);
        let c5 = Float::difference_of_products(self[(2, 2)], self[(3, 3)], self[(3, 2)], self[(2, 3)]);

        Float::difference_of_products(s0, c5, s1, c4)
            + Float::difference_of_products(s2, c3, -s3, c2)
            + Float::difference_of_products(s5, c0, s4, c1)
    }

    #[inline]
    pub fn inverse(self) -> Self {
        self.try_inverse().unwrap()
    }

    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        let s0 = Float::difference_of_products(self[(0, 0)], self[(1, 1)], self[(1, 0)], self[(0, 1)]);
        let s1 = Float::difference_of_products(self[(0, 0)], self[(1, 2)], self[(1, 0)], self[(0, 2)]);
        let s2 = Float::difference_of_products(self[(0, 0)], self[(1, 3)], self[(1, 0)], self[(0, 3)]);

        let s3 = Float::difference_of_products(self[(0, 1)], self[(1, 2)], self[(1, 1)], self[(0, 2)]);
        let s4 = Float::difference_of_products(self[(0, 1)], self[(1, 3)], self[(1, 1)], self[(0, 3)]);
        let s5 = Float::difference_of_products(self[(0, 2)], self[(1, 3)], self[(1, 2)], self[(0, 3)]);

        let c0 = Float::difference_of_products(self[(2, 0)], self[(3, 1)], self[(3, 0)], self[(2, 1)]);
        let c1 = Float::difference_of_products(self[(2, 0)], self[(3, 2)], self[(3, 0)], self[(2, 2)]);
        let c2 = Float::difference_of_products(self[(2, 0)], self[(3, 3)], self[(3, 0)], self[(2, 3)]);

        let c3 = Float::difference_of_products(self[(2, 1)], self[(3, 2)], self[(3, 1)], self[(2, 2)]);
        let c4 = Float::difference_of_products(self[(2, 1)], self[(3, 3)], self[(3, 1)], self[(2, 3)]);
        let c5 = Float::difference_of_products(self[(2, 2)], self[(3, 3)], self[(3, 2)], self[(2, 3)]);

        let det: Float = inner_product(&[s0, -s1, s2, s3, s5, -s4], &[c5, c4, c3, c2, c0, c1]).into();
        if det == 0.0 {
            return None;
        }

        let s = 1.0 / det;

        let inv: [[Float; 4]; 4] = [
            [
                s * Float::from(inner_product(
                    &[self[(1, 1)], self[(1, 3)], -self[(1, 2)]],
                    &[c5, c3, c4],
                )),
                s * Float::from(inner_product(
                    &[-self[(0, 1)], self[(0, 2)], -self[(0, 3)]],
                    &[c5, c4, c3],
                )),
                s * Float::from(inner_product(
                    &[self[(3, 1)], self[(3, 3)], -self[(3, 2)]],
                    &[s5, s3, s4],
                )),
                s * Float::from(inner_product(
                    &[-self[(2, 1)], self[(2, 2)], -self[(2, 3)]],
                    &[s5, s4, s3],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[-self[(1, 0)], self[(1, 2)], -self[(1, 3)]],
                    &[c5, c2, c1],
                )),
                s * Float::from(inner_product(
                    &[self[(0, 0)], self[(0, 3)], -self[(0, 2)]],
                    &[c5, c1, c2],
                )),
                s * Float::from(inner_product(
                    &[-self[(3, 0)], self[(3, 2)], -self[(3, 3)]],
                    &[s5, s2, s1],
                )),
                s * Float::from(inner_product(
                    &[self[(2, 0)], self[(2, 3)], -self[(2, 2)]],
                    &[s5, s1, s2],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[self[(1, 0)], self[(1, 3)], -self[(1, 1)]],
                    &[c4, c0, c2],
                )),
                s * Float::from(inner_product(
                    &[-self[(0, 0)], self[(0, 1)], -self[(0, 3)]],
                    &[c4, c2, c0],
                )),
                s * Float::from(inner_product(
                    &[self[(3, 0)], self[(3, 3)], -self[(3, 1)]],
                    &[s4, s0, s2],
                )),
                s * Float::from(inner_product(
                    &[-self[(2, 0)], self[(2, 1)], -self[(2, 3)]],
                    &[s4, s2, s0],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[-self[(1, 0)], self[(1, 1)], -self[(1, 2)]],
                    &[c3, c1, c0],
                )),
                s * Float::from(inner_product(
                    &[self[(0, 0)], self[(0, 2)], -self[(0, 1)]],
                    &[c3, c0, c1],
                )),
                s * Float::from(inner_product(
                    &[-self[(3, 0)], self[(3, 1)], -self[(3, 2)]],
                    &[s3, s1, s0],
                )),
                s * Float::from(inner_product(
                    &[self[(2, 0)], self[(2, 2)], -self[(2, 1)]],
                    &[s3, s0, s1],
                )),
            ],
        ];

        Some(Mat4::from(inv))
    }
}


impl<T> TMat4<T>
where 
    T: Clone + Copy + NumericConsts + Neg<Output = T> + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T>
{
    pub fn from_translation(delta: Point3<T>) -> Self {
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

    pub fn from_rotation(sin_theta: T, cos_theta: T, axis: Vec3<T>) -> Self {
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

    pub fn from_scale(s: Vec3<T>) -> Self {
        Self::new(
            s.x, T::ZERO, T::ZERO, T::ZERO,
            T::ZERO, s.y, T::ZERO, T::ZERO,
            T::ZERO, T::ZERO, s.z, T::ZERO,
            T::ZERO, T::ZERO, T::ZERO, T::ONE,
        )
    }
}

pub fn mul_mat_vec<const N: usize, V, VResult, M>(m: &M, v: &V) -> VResult
where
    V: Index<usize, Output = Float>,
    VResult: IndexMut<usize, Output = Float> + Default,
    M: Index<(usize, usize), Output = Float>
{
    let mut out: VResult = Default::default();
    for i in 0..N {
        for j in 0..N {
            out[i] += m[(i, j)] * v[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use glam::Vec4Swizzles;
    use mat::mul_mat_vec;

    use crate::math::*;

    #[test]
    fn translate() {
        let m = Mat4::from_translation(Point3f::new(1.0, 2.0, 3.0));
        let p = Point4f::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!((m * p).xyz(), Point3f::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn rotate_axis() {
        let theta = 1.214 as Float;
        let m = Mat4::from_rotation_x(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_x(theta as f32);

        let p = Point4f::new(0.0, 1.0, 0.0, 1.0);
        let gp = glam::Vec4::new(0.0, 1.0, 0.0, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float));

        let theta = 2.3542 as Float;
        let m = Mat4::from_rotation_y(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_y(theta as f32);

        let p = Point4f::new(0.3, 7.1, 3.4, 1.0);
        let gp = glam::Vec4::new(0.3, 7.1, 3.4, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float));

        let theta = 0.761 as Float;
        let m = Mat4::from_rotation_x(theta.sin(), theta.cos());
        let gm = glam::Mat4::from_rotation_x(theta as f32);

        let p = Point4f::new(0.146, 3.0135, 7.0541, 1.0);
        let gp = glam::Vec4::new(0.146, 3.0135, 7.0541, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float));
    }

    #[test]
    fn rotate() {
        let theta = 1.431 as Float;

        let m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(3.0, 1.0, 2.0).normalize());
        let gm = glam::Mat4::from_axis_angle(glam::Vec3::new(3.0, 1.0, 2.0).normalize(), theta as f32);

        let p = Point4f::new(0.413, 4.21, 3.73, 1.0);
        let gp = glam::Vec4::new(0.413, 4.21, 3.73, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float))
    }

    #[test]
    fn scale() {
        let m = Mat4::from_scale(Vec3f::new(1.541, 3.312, 0.231));
        let gm = glam::Mat4::from_scale(glam::Vec3::new(1.541, 3.312, 0.231));

        let p = Point4f::new(1.143, 0.513, 4.041, 1.0);
        let gp = glam::Vec4::new(1.143, 0.513, 4.041, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float))
    }

    #[test]
    fn concat() {
        let theta = 0.712 as Float;
        let mut m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(4.0, 2.0, 5.0).normalize());
        let mut gm = glam::Mat4::from_axis_angle(glam::Vec3::new(4.0, 2.0, 5.0).normalize(), theta as f32);

        m = m * Mat4::from_translation(Point3f::new(3.02, 2.43, 0.42));
        gm = gm.mul_mat4(&glam::Mat4::from_translation(glam::Vec3::new(3.02, 2.43, 0.42)));

        let theta = -1.245 as Float;
        m = m * Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(-2.0, -3.0, 4.0).normalize());
        gm = gm.mul_mat4(&glam::Mat4::from_axis_angle(glam::Vec3::new(-2.0, -3.0, 4.0).normalize(), theta as f32));

        m = m * Mat4::from_scale(Vec3f::new(1.43, 0.53, 2.34));
        gm = gm.mul_mat4(&glam::Mat4::from_scale(glam::Vec3::new(1.43, 0.53, 2.34)));

        let p = Point4f::new(0.51, 2.12, 4.53, 1.0);
        let gp = glam::Vec4::new(0.51, 2.12, 4.53, 1.0);

        let p1 = (m * p).xyz();
        let gp1 = (gm * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float), epsilon = 2e-6 as Float);
    }

    #[test]
    fn determinant() {
        let m = Mat4::from_scale(Vec3f::new(2.0, 3.0, 1.5));
        assert_abs_diff_eq!(m.determinant(), 9.0);
    }

    #[test]
    fn inverse() {
        let theta = -0.834 as Float;
        let m = Mat4::from_rotation(theta.sin(), theta.cos(), Vec3f::new(2.0, 5.0, 7.0).normalize())
            * Mat4::from_translation(Point3f::new(0.9, 0.3, 0.6))
            * Mat4::from_scale(Vec3f::new(0.8, 1.2, 2.3));
        let gm = glam::Mat4::from_axis_angle(glam::Vec3::new(2.0, 5.0, 7.0).normalize(), theta as f32)
            * glam::Mat4::from_translation(glam::Vec3::new(0.9, 0.3, 0.6))
            * glam::Mat4::from_scale(glam::Vec3::new(0.8, 1.2, 2.3));
        
        let p = Point4f::new(2.23, 0.83, 1.74, 1.0);
        let gp = glam::Vec4::new(2.23, 0.83, 1.74, 1.0);

        let p1 = (m.inverse() * p).xyz();
        let gp1 = (gm.inverse() * gp).xyz();

        assert_abs_diff_eq!(p1, Point3f::new(gp1.x as Float, gp1.y as Float, gp1.z as Float));
    }

    #[test]
    fn test_mul_mat_vec() {
        let m = Mat3::new(0.531, -0.61, 0.631, 0.613, 0.657, 0.81, 0.134, 0.246, 0.136);
        let v = Vec3f::new(2.15, 0.53, 3.87);

        assert_abs_diff_eq!(mul_mat_vec::<3, Vec3f, Vec3f, Mat3>(&m, &v), m * v);
    }
}
