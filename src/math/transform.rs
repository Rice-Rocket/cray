// Pbrt 3.9 Transformations

use std::ops::Mul;

use crate::math::*;


// TODO: Test this
pub struct Transform {
    pub m: Mat4,
    m_inv: Option<Mat4>,
}

impl Default for Transform {
    fn default() -> Self {
        Self { m: Mat4::identity(), m_inv: Some(Mat4::identity()) }
    }
}

impl Transform {
    pub fn new(m: Mat4) -> Self {
        Self { m, m_inv: None }
    }

    pub fn new_with_inverse(m: Mat4, m_inv: Mat4) -> Self {
        Self { m, m_inv: Some(m_inv) }
    }

    pub fn from_translation(delta: &Vec3) -> Transform {
        Self {
            m: Mat4::new_translation(&Vec3::new(delta.x, delta.y, delta.z)),
            m_inv: Some(Mat4::new_translation(&Vec3::new(-delta.x, -delta.y, -delta.z))),
        }
    }

    pub fn from_rotation_x(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos_theta, -sin_theta, 0.0,
            0.0, sin_theta, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Self {
            m,
            m_inv: Some(m.transpose())
        }
    }

    pub fn from_rotation_y(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::new(
            cos_theta, 0.0, sin_theta, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin_theta, 0.0, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Self {
            m,
            m_inv: Some(m.transpose())
        }
    }

    pub fn from_rotation_z(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::new(
            cos_theta, -sin_theta, 0.0, 0.0,
            sin_theta, cos_theta, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Self {
            m,
            m_inv: Some(m.transpose())
        }
    }

    pub fn from_rotation(axisangle: Vec3) -> Transform {
        Self { m: Mat4::new_rotation(axisangle), m_inv: Some(Mat4::new_rotation(axisangle).transpose()) }
    }

    pub fn from_delta(from: &Bivec3, to: &Bivec3) -> Transform {
        let refl = if from.x().abs() < 0.72 && to.x().abs() < 0.72 {
            Vec3::new(1.0, 0.0, 0.0)
        } else if from.y().abs() < 0.72 && to.y().abs() < 0.72 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let u = refl - from.get();
        let v = refl - to.get();
        let mut r = Mat4::default();
        for i in 0..3 {
            for j in 0..3 {
                r[(i, j)] = if i == j { 1.0 } else { 0.0 }
                    - 2.0 / u.dot(&u) * u[i] * u[j]
                    - 2.0 / v.dot(&v) * v[i] * v[j]
                    + 4.0 * u.dot(&v) / (u.dot(&u) * v.dot(&v)) * v[i] * u[j];
            }
        }

        Self {
            m: r,
            m_inv: Some(r.transpose())
        }
    }

    pub fn looking_at(pos: &Vec3, target: &Vec3, up: &Bivec3) -> Transform {
        let dir = Bivec3::new_normalize(target - pos);
        let right = Bivec3::new_normalize(up.cross(&dir));
        let new_up = dir.cross(&right);

        let m_inv = Mat4::new(
            right.x(), new_up.x, dir.x(), pos.x,
            right.y(), new_up.y, dir.y(), pos.y, 
            right.z(), new_up.z, dir.z(), pos.z, 
            0.0, 0.0, 0.0, 1.0
        );

        let m = m_inv.try_inverse().unwrap();
        Self { m, m_inv: Some(m_inv) }
    }

    pub fn from_scale(scale: &Vec3) -> Transform {
        Self {
            m: Mat4::new_nonuniform_scaling(&Vec3::new(scale.x, scale.y, scale.z)),
            m_inv: Some(Mat4::new_nonuniform_scaling(&Vec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z))),
        }
    }

    pub fn inverse(&mut self) -> Mat4 {
        self.try_inverse().expect("Matrix transform is not invertible")
    }

    pub fn try_inverse(&mut self) -> Option<Mat4> {
        match self.m_inv {
            Some(inv) => Some(inv),
            None => {
                self.m_inv = self.m.try_inverse();
                self.m_inv
            },
        }
    }

    pub fn transpose(&mut self) {
        self.m.transpose_mut();
        if let Some(inv) = self.m_inv.as_mut() {
            inv.transpose_mut();
        }
    }

    pub fn transposed(&self) -> Transform {
        Self { m: self.m.transpose(), m_inv: self.m_inv.map(|inv| inv.transpose()) }
    }

    pub fn is_identity(&self) -> bool {
        self.m.is_identity(Scalar::EPSILON)
    }

    pub fn has_scale(&self, epsilon: Scalar) -> bool {
        let la2 = (self * Vec3::new(1.0, 0.0, 0.0)).magnitude_squared();
        let lb2 = (self * Vec3::new(0.0, 1.0, 0.0)).magnitude_squared();
        let lc2 = (self * Vec3::new(0.0, 0.0, 1.0)).magnitude_squared();
        (la2 - 1.0).abs() > epsilon || (lb2 - 1.0).abs() > epsilon || (lc2 - 1.0).abs() > epsilon
    }
}

impl Mul<&Vec3> for &Transform {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        (self.m * Vec4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<&Vec4> for &Transform {
    type Output = Vec4;

    fn mul(self, rhs: &Vec4) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<&Vec3> for Transform {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        (self.m * Vec4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<&Vec4> for Transform {
    type Output = Vec4;

    fn mul(self, rhs: &Vec4) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<Vec3> for &Transform {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        (self.m * Vec4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<Vec4> for &Transform {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<Vec3> for Transform {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        (self.m * Vec4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<Vec4> for Transform {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        self.m * rhs
    }
}

impl PartialEq for Transform {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
    }
}
