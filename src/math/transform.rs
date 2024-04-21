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
        Self { m: Mat4::IDENTITY, m_inv: Some(Mat4::IDENTITY) }
    }
}

impl Transform {
    pub fn new(m: Mat4) -> Self {
        Self { m, m_inv: None }
    }

    pub fn new_with_inverse(m: Mat4, m_inv: Mat4) -> Self {
        Self { m, m_inv: Some(m_inv) }
    }

    pub fn from_translation(delta: Point3) -> Transform {
        Transform {
            m: Mat4::from_translation(delta),
            m_inv: Some(Mat4::from_translation(-delta)),
        }
    }

    pub fn from_rotation_x(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_x(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: Some(m.transpose()),
        }
    }

    pub fn from_rotation_y(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_y(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: Some(m.transpose())
        }
    }

    pub fn from_rotation_z(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_z(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: Some(m.transpose())
        }
    }

    pub fn from_rotation(sin_theta: Scalar, cos_theta: Scalar, axis: UnitVec3) -> Transform {
        let m = Mat4::from_rotation(sin_theta, cos_theta, axis);
        Transform { m, m_inv: Some(m.transpose()) }
    }

    pub fn from_rotation_delta(from: UnitVec3, to: UnitVec3) -> Transform {
        let refl = if from.x.abs() < 0.72 && to.x.abs() < 0.72 {
            Point3::new(1.0, 0.0, 0.0)
        } else if from.y.abs() < 0.72 && to.y.abs() < 0.72 {
            Point3::new(0.0, 1.0, 0.0)
        } else {
            Point3::new(0.0, 0.0, 1.0)
        };

        let u = refl - from;
        let v = refl - to;
        let mut r = Mat4::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                r[(i, j)] = if i == j { 1.0 } else { 0.0 }
                    - 2.0 / u.dot(u) * u[i] * u[j]
                    - 2.0 / v.dot(v) * v[i] * v[j]
                    + 4.0 * u.dot(v) / (u.dot(u) * v.dot(v)) * v[i] * u[j];
            }
        }

        Transform {
            m: r,
            m_inv: Some(r.transpose())
        }
    }

    pub fn looking_at(pos: Point3, target: Point3, up: UnitVec3) -> Transform {
        let dir = UnitVec3::from((target - pos).normalize());
        let right = UnitVec3::from(up.cross(dir).normalize());
        let new_up = dir.cross(right);

        let m_inv = Mat4::new(
            right.x, new_up.x, dir.x, pos.x,
            right.y, new_up.y, dir.y, pos.y, 
            right.z, new_up.z, dir.z, pos.z, 
            0.0, 0.0, 0.0, 1.0
        );

        let m = m_inv.inverse();
        Transform { m, m_inv: Some(m_inv) }
    }

    pub fn from_scale(scale: Vec3) -> Transform {
        Transform {
            m: Mat4::from_scale(scale),
            m_inv: Some(Mat4::from_scale(scale.recip())),
        }
    }

    pub fn inverse(&mut self) -> Mat4 {
        self.try_inverse().expect("Matrix transform is not invertible")
    }

    pub fn try_inverse(&mut self) -> Option<Mat4> {
        match self.m_inv {
            Some(inv) => Some(inv),
            None => {
                self.m_inv = Some(self.m.inverse());
                self.m_inv
            },
        }
    }

    pub fn transpose(self) -> Transform {
        Transform {
            m: self.m.transpose(),
            m_inv: self.m_inv.map(|m| m.transpose())
        }
    }

    pub fn is_identity(&self) -> bool {
        self.m == Mat4::IDENTITY
    }

    pub fn has_scale(&self, epsilon: Scalar) -> bool {
        let la2 = (self * Point3::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = (self * Point3::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = (self * Point3::new(0.0, 0.0, 1.0)).length_squared();
        (la2 - 1.0).abs() > epsilon || (lb2 - 1.0).abs() > epsilon || (lc2 - 1.0).abs() > epsilon
    }
}

impl Mul<Vec3> for &Transform {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<Vec3> for Transform {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<UnitVec3> for &Transform {
    type Output = Vec3;

    fn mul(self, rhs: UnitVec3) -> Self::Output {
        Vec3::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<UnitVec3> for Transform {
    type Output = Vec3;

    fn mul(self, rhs: UnitVec3) -> Self::Output {
        Vec3::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<Point3> for &Transform {
    type Output = Point3;

    fn mul(self, rhs: Point3) -> Self::Output {
        (self.m * Point4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<Point4> for &Transform {
    type Output = Point4;

    fn mul(self, rhs: Point4) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<Point3> for Transform {
    type Output = Point3;

    fn mul(self, rhs: Point3) -> Self::Output {
        (self.m * Point4::new(rhs.x, rhs.y, rhs.z, 1.0)).xyz()
    }
}

impl Mul<Point4> for Transform {
    type Output = Point4;

    fn mul(self, rhs: Point4) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Self::Output {
        Transform {
            m: self.m * rhs.m,
            m_inv: self.m_inv.zip(rhs.m_inv).map(|(a, b)| a * b),
        }
    }
}

impl PartialEq for Transform {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
    }
}
