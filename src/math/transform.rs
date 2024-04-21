// Pbrt 3.9 Transformations

use std::ops::Mul;

use crate::math::*;


// TODO: Test this
#[derive(Clone, Copy)]
pub struct Transform {
    pub m: Mat4,
    pub m_inv: Mat4,
}

impl Default for Transform {
    fn default() -> Self {
        Self { m: Mat4::IDENTITY, m_inv: Mat4::IDENTITY }
    }
}

impl Transform {
    #[inline]
    pub fn new(m: Mat4, m_inv: Mat4) -> Self {
        Self { m, m_inv }
    }

    #[inline]
    pub fn new_with_inverse(m: Mat4) -> Self {
        Self { m, m_inv: m.inverse() }
    }

    #[inline]
    pub fn from_translation(delta: Point3f) -> Transform {
        Transform {
            m: Mat4::from_translation(delta),
            m_inv: Mat4::from_translation(-delta),
        }
    }

    #[inline]
    pub fn from_rotation_x(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_x(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose(),
        }
    }

    #[inline]
    pub fn from_rotation_y(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_y(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose()
        }
    }

    #[inline]
    pub fn from_rotation_z(sin_theta: Scalar, cos_theta: Scalar) -> Transform {
        let m = Mat4::from_rotation_z(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose()
        }
    }

    #[inline]
    pub fn from_rotation(sin_theta: Scalar, cos_theta: Scalar, axis: UnitVec3f) -> Transform {
        let m = Mat4::from_rotation(sin_theta, cos_theta, axis);
        Transform { m, m_inv: m.transpose() }
    }

    #[inline]
    pub fn from_rotation_delta(from: UnitVec3f, to: UnitVec3f) -> Transform {
        let refl = if from.x.abs() < 0.72 && to.x.abs() < 0.72 {
            Point3f::new(1.0, 0.0, 0.0)
        } else if from.y.abs() < 0.72 && to.y.abs() < 0.72 {
            Point3f::new(0.0, 1.0, 0.0)
        } else {
            Point3f::new(0.0, 0.0, 1.0)
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
            m_inv: r.transpose()
        }
    }

    #[inline]
    pub fn looking_at(pos: Point3f, target: Point3f, up: UnitVec3f) -> Transform {
        let dir = UnitVec3f::from((target - pos).normalize());
        let right = UnitVec3f::from(up.cross(dir).normalize());
        let new_up = dir.cross(right);

        let m_inv = Mat4::new(
            right.x, new_up.x, dir.x, pos.x,
            right.y, new_up.y, dir.y, pos.y, 
            right.z, new_up.z, dir.z, pos.z, 
            0.0, 0.0, 0.0, 1.0
        );

        let m = m_inv.inverse();
        Transform { m, m_inv }
    }

    #[inline]
    pub fn from_scale(scale: Vec3f) -> Transform {
        Transform {
            m: Mat4::from_scale(scale),
            m_inv: Mat4::from_scale(scale.recip()),
        }
    }

    #[inline]
    pub fn from_frame(frame: Frame) -> Transform {
        let m = Mat4::new(
            frame.x.x, frame.x.y, frame.x.z, 0.0,
            frame.y.x, frame.y.y, frame.y.z, 0.0,
            frame.z.x, frame.z.y, frame.z.z, 0.0,
            0.0,       0.0,       0.0,       1.0,
        );
        Self::new_with_inverse(m)
    }

    #[inline]
    pub fn inverse(&self) -> Transform {
        Transform {
            m: self.m_inv,
            m_inv: self.m,
        }
    }

    #[inline]
    pub fn transpose(self) -> Transform {
        Transform {
            m: self.m.transpose(),
            m_inv: self.m_inv.transpose()
        }
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.m == Mat4::IDENTITY
    }

    #[inline]
    pub fn has_scale(self, epsilon: Scalar) -> bool {
        let la2 = (self * Point3f::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = (self * Point3f::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = (self * Point3f::new(0.0, 0.0, 1.0)).length_squared();
        (la2 - 1.0).abs() > epsilon || (lb2 - 1.0).abs() > epsilon || (lc2 - 1.0).abs() > epsilon
    }

    #[inline]
    pub fn swaps_handedness(&self) -> bool {
        let m = Mat3::new(
            self.m[(0, 0)], self.m[(0, 1)], self.m[(0, 2)],
            self.m[(1, 0)], self.m[(1, 1)], self.m[(1, 2)],
            self.m[(2, 0)], self.m[(2, 1)], self.m[(2, 2)],
        );
        m.determinant() < 0.0
    }
}

impl Mul<Vec3f> for Transform {
    type Output = Vec3f;

    #[inline]
    fn mul(self, rhs: Vec3f) -> Self::Output {
        Vec3f::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<UnitVec3f> for Transform {
    type Output = Vec3f;

    #[inline]
    fn mul(self, rhs: UnitVec3f) -> Self::Output {
        Vec3f::new(
            self.m[(0, 0)] * rhs.x + self.m[(0, 1)] * rhs.y + self.m[(0, 2)] * rhs.z,
            self.m[(1, 0)] * rhs.x + self.m[(1, 1)] * rhs.y + self.m[(1, 2)] * rhs.z,
            self.m[(2, 0)] * rhs.x + self.m[(2, 1)] * rhs.y + self.m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<Point3f> for Transform {
    type Output = Point3f;

    #[inline]
    fn mul(self, rhs: Point3f) -> Self::Output {
        let p = self.m * Point4f::new(rhs.x, rhs.y, rhs.z, 1.0);
        p.xyz() / p.w
    }
}

impl Mul<Point4f> for Transform {
    type Output = Point4f;

    fn mul(self, rhs: Point4f) -> Self::Output {
        self.m * rhs
    }
}

impl Mul<Point3fi> for Transform {
    type Output = Point3fi;

    fn mul(self, rhs: Point3fi) -> Self::Output {
        let p = Mat4i::from(self.m) * Point4fi::new(rhs.x, rhs.y, rhs.z, Interval::from_val(1.0));
        p.xyz() / p.w
    }
}

impl Mul<Point4fi> for Transform {
    type Output = Point4fi;

    #[inline]
    fn mul(self, rhs: Point4fi) -> Self::Output {
        Mat4i::from(self.m) * rhs
    }
}

impl Mul<Normal3f> for Transform {
    type Output = Normal3f;

    #[inline]
    fn mul(self, rhs: Normal3f) -> Self::Output {
        let m = self.m_inv;
        Normal3f::new_normalize(
            m[(0, 0)] * rhs.x + m[(1, 0)] * rhs.y + m[(2, 0)] * rhs.z,
            m[(0, 1)] * rhs.x + m[(1, 1)] * rhs.y + m[(2, 1)] * rhs.z,
            m[(0, 2)] * rhs.x + m[(1, 2)] * rhs.y + m[(2, 2)] * rhs.z,
        )
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    #[inline]
    fn mul(self, rhs: Transform) -> Self::Output {
        Transform {
            m: self.m * rhs.m,
            m_inv: self.m_inv * rhs.m_inv,
        }
    }
}

impl Mul<Ray> for Transform {
    type Output = Ray;

    #[inline]
    fn mul(self, rhs: Ray) -> Self::Output {
        let mut o = self * Point3fi::from(rhs.origin);
        let d = self * rhs.direction;
        
        let length_sqr = d.length_squared();
        if length_sqr > 0.0 {
            let dt = d.abs().dot(o.error().into()) / length_sqr;
            o = o + TVec3::from(d * dt);
        }

        Ray::new(o.into(), d.into())
    }
}

impl Mul<RayDifferential> for Transform {
    type Output = RayDifferential;

    fn mul(self, rhs: RayDifferential) -> Self::Output {
        let tr = self * rhs.ray;

        let aux: Option<AuxiliaryRays> = if let Some(aux) = &rhs.aux {
            let rx_origin = self * aux.rx_origin;
            let rx_direction = self * aux.rx_direction;
            let ry_origin = self * aux.ry_origin;
            let ry_direction = self * aux.ry_direction;
            Some(AuxiliaryRays::new(
                rx_origin,
                rx_direction.into(),
                ry_origin,
                ry_direction.into(),
            ))
        } else {
            None
        };

        RayDifferential { ray: tr, aux }
    }
}

impl Mul<Bounds3f> for Transform {
    type Output = Bounds3f;

    #[inline]
    fn mul(self, rhs: Bounds3f) -> Self::Output {
        let mut b = Bounds3f::new();
        for i in 0..8 {
            b |= self * rhs.corner(i);
        }
        b
    }
}

impl PartialEq for Transform {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
    }
}
