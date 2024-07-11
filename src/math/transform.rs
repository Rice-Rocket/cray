// Pbrt 3.9 Transformations + 3.10 Applying Transformations

use std::ops::Mul;

use interaction::{Interaction, SurfaceInteraction, SurfaceInteractionShading};

use crate::math::*;


#[derive(Debug, Clone, Copy)]
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
    pub fn from_rotation_x(sin_theta: Float, cos_theta: Float) -> Transform {
        let m = Mat4::from_rotation_x(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose(),
        }
    }

    #[inline]
    pub fn from_rotation_y(sin_theta: Float, cos_theta: Float) -> Transform {
        let m = Mat4::from_rotation_y(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose()
        }
    }

    #[inline]
    pub fn from_rotation_z(sin_theta: Float, cos_theta: Float) -> Transform {
        let m = Mat4::from_rotation_z(sin_theta, cos_theta);
        Transform {
            m,
            m_inv: m.transpose()
        }
    }

    #[inline]
    pub fn from_rotation(sin_theta: Float, cos_theta: Float, axis: Vec3f) -> Transform {
        debug_assert!(axis.is_normalized());

        let m = Mat4::from_rotation(sin_theta, cos_theta, axis);
        Transform { m, m_inv: m.transpose() }
    }

    #[inline]
    pub fn from_rotation_delta(from: Vec3f, to: Vec3f) -> Transform {
        debug_assert!(from.is_normalized());
        debug_assert!(to.is_normalized());

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
    pub fn looking_at(pos: Point3f, target: Point3f, up: Vec3f) -> Transform {
        debug_assert!(up.is_normalized());

        let dir = (target - pos).normalize().into();
        let right = up.cross(dir).normalize();
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
    pub fn has_scale(self, epsilon: Float) -> bool {
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

    #[inline]
    pub fn orthographic(z_near: Float, z_far: Float) -> Transform {
        Transform::from_scale(Vec3f::new(1.0, 1.0, 1.0 / (z_far - z_near)))
            * Transform::from_translation(Point3f::new(0.0, 0.0, -z_near))
    }

    #[inline]
    pub fn perspective(fov: Float, n: Float, f: Float) -> Transform {
        let per = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, f / (f - n), -f * n / (f - n),
            0.0, 0.0, 1.0, 0.0,
        );

        let inv_tan_ang = 1.0 / Float::tan(to_radians(fov) / 2.0);
        Transform::from_scale(Vec3f::new(inv_tan_ang, inv_tan_ang, 1.0)) * Transform::new_with_inverse(per)
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
        let x: Float = rhs.x.into();
        let y: Float = rhs.y.into();
        let z: Float = rhs.z.into();
        // Compute transformed coordinates
        let xp: Float = (self.m[(0, 0)] * x + self.m[(0, 1)] * y)
            + (self.m[(0, 2)] * z + self.m[(0, 3)]);
        let yp: Float = (self.m[(1, 0)] * x + self.m[(1, 1)] * y)
            + (self.m[(1, 2)] * z + self.m[(1, 3)]);
        let zp: Float = (self.m[(2, 0)] * x + self.m[(2, 1)] * y)
            + (self.m[(2, 2)] * z + self.m[(2, 3)]);
        let wp: Float = (self.m[(3, 0)] * x + self.m[(3, 1)] * y)
            + (self.m[(3, 2)] * z + self.m[(3, 3)]);

        // Compute absolute error for transformed point
        let p_error: Vec3f = if rhs.is_exact() {
            // Compute error for transformed exact _p_
            let err_x = gamma(3)
                * (Float::abs(self.m[(0, 0)] * x)
                    + Float::abs(self.m[(0, 1)] * y)
                    + Float::abs(self.m[(0, 2)] * z)
                    + Float::abs(self.m[(0, 3)]));
            let err_y = gamma(3)
                * (Float::abs(self.m[(1, 0)] * x)
                    + Float::abs(self.m[(1, 1)] * y)
                    + Float::abs(self.m[(1, 2)] * z)
                    + Float::abs(self.m[(1, 3)]));
            let err_z = gamma(3)
                * (Float::abs(self.m[(2, 0)] * x)
                    + Float::abs(self.m[(2, 1)] * y)
                    + Float::abs(self.m[(2, 2)] * z)
                    + Float::abs(self.m[(2, 3)]));
            Vec3f::new(err_x, err_y, err_z)
        } else {
            // Compute error for transformed approximate _p_
            let p_in_error = rhs.error();
            let err_x = (gamma(3) + 1.0)
                * (Float::abs(self.m[(0, 0)]) * p_in_error.x
                    + Float::abs(self.m[(0, 1)]) * p_in_error.y
                    + Float::abs(self.m[(0, 2)]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(0, 0)] * x)
                        + Float::abs(self.m[(0, 1)] * y)
                        + Float::abs(self.m[(0, 2)] * z)
                        + Float::abs(self.m[(0, 3)]));
            let err_y = (gamma(3) + 1.0)
                * (Float::abs(self.m[(1, 0)]) * p_in_error.x
                    + Float::abs(self.m[(1, 1)]) * p_in_error.y
                    + Float::abs(self.m[(1, 2)]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(1, 0)] * x)
                        + Float::abs(self.m[(1, 1)] * y)
                        + Float::abs(self.m[(1, 2)] * z)
                        + Float::abs(self.m[(1, 3)]));
            let err_z = (gamma(3) + 1.0)
                * (Float::abs(self.m[(2, 0)]) * p_in_error.x
                    + Float::abs(self.m[(2, 1)]) * p_in_error.y
                    + Float::abs(self.m[(2, 2)]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(2, 0)] * x)
                        + Float::abs(self.m[(2, 1)] * y)
                        + Float::abs(self.m[(2, 2)] * z)
                        + Float::abs(self.m[(2, 3)]));
            Vec3f::new(err_x, err_y, err_z)
        };
        if wp == 1.0 {
            Point3fi::from_errors(Point3f::new(xp, yp, zp), p_error.into())
        } else {
            Point3fi::from_errors(Point3f::new(xp, yp, zp), p_error.into()) / Interval::from(wp)
        }
    }
}

impl Mul<Vec3fi> for Transform {
    type Output = Vec3fi;

    fn mul(self, rhs: Vec3fi) -> Self::Output {
        let x: Float = rhs.x.into();
        let y: Float = rhs.y.into();
        let z: Float = rhs.z.into();
        let v_out_err = if rhs.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(self.m[(0, 0)] * x)
                    + Float::abs(self.m[(0, 1)] * y)
                    + Float::abs(self.m[(0, 2)] * z));
            let y_err = gamma(3)
                * (Float::abs(self.m[(1, 0)] * x)
                    + Float::abs(self.m[(1, 1)] * y)
                    + Float::abs(self.m[(1, 2)] * z));
            let z_err = gamma(3)
                * (Float::abs(self.m[(2, 0)] * x)
                    + Float::abs(self.m[(2, 1)] * y)
                    + Float::abs(self.m[(2, 2)] * z));
            Vec3f::new(x_err, y_err, z_err)
        } else {
            let v_in_error = rhs.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[(0, 0)]) * v_in_error.x
                    + Float::abs(self.m[(0, 1)]) * v_in_error.y
                    + Float::abs(self.m[(0, 2)]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(0, 0)] * x)
                        + Float::abs(self.m[(0, 1)] * y)
                        + Float::abs(self.m[(0, 2)] * z));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[(1, 0)]) * v_in_error.x
                    + Float::abs(self.m[(1, 1)]) * v_in_error.y
                    + Float::abs(self.m[(1, 2)]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(1, 0)] * x)
                        + Float::abs(self.m[(1, 1)] * y)
                        + Float::abs(self.m[(1, 2)] * z));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[(2, 0)]) * v_in_error.x
                    + Float::abs(self.m[(2, 1)]) * v_in_error.y
                    + Float::abs(self.m[(2, 2)]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[(2, 0)] * x)
                        + Float::abs(self.m[(2, 1)] * y)
                        + Float::abs(self.m[(2, 2)] * z));
            Vec3f::new(x_err, y_err, z_err)
        };

        let xp: Float = self.m[(0, 0)] * x + self.m[(0, 1)] * y + self.m[(0, 2)] * z;
        let yp: Float = self.m[(1, 0)] * x + self.m[(1, 1)] * y + self.m[(1, 2)] * z;
        let zp: Float = self.m[(2, 0)] * x + self.m[(2, 1)] * y + self.m[(2, 2)] * z;

        Vec3fi::from_errors(Vec3f::new(xp, yp, zp), v_out_err)
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
        Normal3f::new(
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

        Ray::new(o.into(), d)
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
                rx_direction,
                ry_origin,
                ry_direction,
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
        let mut b = Bounds3f::default();
        for i in 0..8 {
            b |= self * rhs.corner(i);
        }
        b
    }
}

impl Mul<SurfaceInteraction> for Transform {
    type Output = SurfaceInteraction;

    fn mul(self, rhs: SurfaceInteraction) -> Self::Output {
        let t = self.inverse();

        let n = (t * rhs.interaction.n).normalize();

        SurfaceInteraction {
            interaction: Interaction {
                pi: self * rhs.interaction.pi,
                time: rhs.interaction.time,
                wo: (t * rhs.interaction.wo).normalize(),
                n,
                uv: rhs.interaction.uv
            },
            dpdu: t * rhs.dpdu,
            dpdv: t * rhs.dpdv,
            dndu: t * rhs.dndu,
            dndv: t * rhs.dndv,
            shading: SurfaceInteractionShading {
                n: (t * rhs.shading.n).normalize().facing(n),
                dpdu: t * rhs.shading.dpdu,
                dpdv: t * rhs.shading.dpdv,
                dndu: t * rhs.shading.dndu,
                dndv: t * rhs.shading.dndv,
            },
            face_index: rhs.face_index,
            material: rhs.material.clone(),
            area_light: rhs.area_light.clone(),
            dpdx: t * rhs.dpdx,
            dpdy: t * rhs.dpdy,
            dudx: rhs.dudx,
            dvdx: rhs.dvdx,
            dudy: rhs.dudy,
            dvdy: rhs.dvdy,
        }
    }
}

impl PartialEq for Transform {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn scale_normal() {
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::from_scale(Vec3f::new(2.0, 3.0, 4.0));
        let scaled = scale * p;
        assert_eq!(Normal3f::new(0.5, 0.6666667, 0.75), scaled);
        let back_again = scale.inverse() * scaled;
        assert_eq!(p, back_again);

        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::from_scale(Vec3f::new(2.0, 2.0, 2.0));
        let scaled = scale * p;
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), scaled);
        let back_again = scale.inverse() * scaled;
        assert_eq!(p, back_again);
    }

    #[test]
    fn bb_transform() {
        let b = Bounds3f::from_points(vec![Point3f::ZERO, Point3f::ONE]);
        let m = Transform::from_translation(Point3f::ONE);
        assert_eq!(Bounds3f::from_points(vec![Point3f::ONE, Point3f::ONE * 2.0]), m * b);
    }

    #[test]
    fn rotate_from_to() {
        let from = Vec3f::new(2.0, 4.0, 1.0).normalize();
        let to = Vec3f::new(3.0, 1.0, 4.0).normalize();
        let r = Transform::from_rotation_delta(from, to);
        assert_abs_diff_eq!(to, r * from, epsilon = 2e-6);

        let from = Vec3f::new(-1.0, -5.0, 3.0).normalize();
        let to = Vec3f::new(3.0, 1.0, -2.0).normalize();
        let r = Transform::from_rotation_delta(from, to);
        assert_abs_diff_eq!(to, r * from, epsilon = 2e-6);
    }
}
