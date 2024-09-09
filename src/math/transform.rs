// Pbrt 3.9 Transformations + 3.10 Applying Transformations

use std::ops::Mul;

use bounds::Union;
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
        let la2 = (self.apply(Point3f::new(1.0, 0.0, 0.0))).length_squared();
        let lb2 = (self.apply(Point3f::new(0.0, 1.0, 0.0))).length_squared();
        let lc2 = (self.apply(Point3f::new(0.0, 0.0, 1.0))).length_squared();
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
            .apply(Transform::from_translation(Point3f::new(0.0, 0.0, -z_near)))
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
        Transform::from_scale(Vec3f::new(inv_tan_ang, inv_tan_ang, 1.0)).apply(Transform::new_with_inverse(per))
    }
}

pub trait ApplyTransform<T> {
    fn apply(&self, rhs: T) -> T;
}

pub trait ApplyRayTransform<T> {
    fn apply_ray(&self, ray: &T, t_max: Option<&mut Float>) -> T;
}

pub trait ApplyInverseTransform<T> {
    fn apply_inverse(&self, rhs: T) -> T;
}

pub trait ApplyRayInverseTransform<T> {
    fn apply_ray_inverse(&self, ray: &T, t_max: Option<&mut Float>) -> T;
}

impl ApplyTransform<Point3f> for Transform {
    fn apply(&self, rhs: Point3f) -> Point3f {
        apply_point(&self.m, &rhs)
    }
}

impl ApplyTransform<Vec3f> for Transform {
    fn apply(&self, rhs: Vec3f) -> Vec3f {
        apply_vector(&self.m, &rhs)
    }
}

impl ApplyTransform<Normal3f> for Transform {
    fn apply(&self, rhs: Normal3f) -> Normal3f {
        // Normals are transformed by the inverse
        apply_normal(&self.m_inv, &rhs)
    }
}

impl ApplyTransform<Point3fi> for Transform {
    fn apply(&self, rhs: Point3fi) -> Point3fi {
        let x: Float = rhs.x.into();
        let y: Float = rhs.y.into();
        let z: Float = rhs.z.into();

        let xp: Float = (self.m[(0, 0)] * x + self.m[(0, 1)] * y)
            + (self.m[(0, 2)] * z + self.m[(0, 3)]);
        let yp: Float = (self.m[(1, 0)] * x + self.m[(1, 1)] * y)
            + (self.m[(1, 2)] * z + self.m[(1, 3)]);
        let zp: Float = (self.m[(2, 0)] * x + self.m[(2, 1)] * y)
            + (self.m[(2, 2)] * z + self.m[(2, 3)]);
        let wp: Float = (self.m[(3, 0)] * x + self.m[(3, 1)] * y)
            + (self.m[(3, 2)] * z + self.m[(3, 3)]);

        let p_error: Vec3f = if rhs.is_exact() {
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

impl ApplyTransform<Vec3fi> for Transform {
    fn apply(&self, rhs: Vec3fi) -> Vec3fi {
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

impl ApplyRayTransform<Ray> for Transform {
    fn apply_ray(&self, ray: &Ray, t_max: Option<&mut Float>) -> Ray {
        let o: Point3fi = self.apply(ray.origin).into();
        let d: Vec3f = self.apply(ray.direction);
        let length_squared = d.length_squared();
        let o: Point3fi = if length_squared > 0.0 {
            let dt = d.abs().dot(o.error()) / length_squared;
            if let Some(t_max) = t_max {
                *t_max -= dt;
            }
            o + Vec3fi::from(d * dt)
        } else {
            o
        };
        Ray::new_with_medium_time(o.into(), d, ray.time, ray.medium.clone())
    }
}

impl ApplyRayTransform<RayDifferential> for Transform {
    fn apply_ray(&self, ray: &RayDifferential, t_max: Option<&mut Float>) -> RayDifferential {
        let tr = self.apply_ray(&ray.ray, t_max);
        let aux: Option<AuxiliaryRays> = if let Some(aux) = &ray.aux {
            let rx_origin = self.apply(aux.rx_origin);
            let rx_direction = self.apply(aux.rx_direction);
            let ry_origin = self.apply(aux.ry_origin);
            let ry_direction = self.apply(aux.ry_direction);
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

impl ApplyTransform<Bounds3f> for Transform {
    fn apply(&self, rhs: Bounds3f) -> Bounds3f {
        let mut out = Bounds3f::new(
            self.apply(rhs.corner(0)),
            self.apply(rhs.corner(1)),
        );

        for i in 2..8 {
            out = out.union(self.apply(rhs.corner(i)));
        }

        out
    }
}

impl ApplyTransform<SurfaceInteraction> for Transform {
    fn apply(&self, rhs: SurfaceInteraction) -> SurfaceInteraction {
        let t = self.inverse();

        let n = t.apply(rhs.interaction.n).normalize();

        SurfaceInteraction {
            interaction: Interaction {
                pi: self.apply(rhs.interaction.pi),
                time: rhs.interaction.time,
                wo: t.apply(rhs.interaction.wo).normalize(),
                n,
                uv: rhs.interaction.uv,
                medium: rhs.interaction.medium,
                medium_interface: rhs.interaction.medium_interface,
            },
            dpdu: t.apply(rhs.dpdu),
            dpdv: t.apply(rhs.dpdv),
            dndu: t.apply(rhs.dndu),
            dndv: t.apply(rhs.dndv),
            shading: SurfaceInteractionShading {
                n: t.apply(rhs.shading.n).normalize().facing(n),
                dpdu: t.apply(rhs.shading.dpdu),
                dpdv: t.apply(rhs.shading.dpdv),
                dndu: t.apply(rhs.shading.dndu),
                dndv: t.apply(rhs.shading.dndv),
            },
            face_index: rhs.face_index,
            material: rhs.material.clone(),
            area_light: rhs.area_light.clone(),
            dpdx: t.apply(rhs.dpdx),
            dpdy: t.apply(rhs.dpdy),
            dudx: rhs.dudx,
            dvdx: rhs.dvdx,
            dudy: rhs.dudy,
            dvdy: rhs.dvdy,
        }
    }
}

impl ApplyInverseTransform<Point3f> for Transform {
    fn apply_inverse(&self, rhs: Point3f) -> Point3f {
        apply_point(&self.m_inv, &rhs)
    }
}

impl ApplyInverseTransform<Vec3f> for Transform {
    fn apply_inverse(&self, rhs: Vec3f) -> Vec3f {
        apply_vector(&self.m_inv, &rhs)
    }
}

impl ApplyInverseTransform<Normal3f> for Transform {
    fn apply_inverse(&self, rhs: Normal3f) -> Normal3f {
        // Normals are transformed by the inverse
        apply_normal(&self.m, &rhs)
    }
}

impl ApplyInverseTransform<Point3fi> for Transform {
    fn apply_inverse(&self, rhs: Point3fi) -> Point3fi {
        let x: Float = rhs.x.into();
        let y: Float = rhs.y.into();
        let z: Float = rhs.z.into();

        let xp: Float = (self.m_inv[(0, 0)] * x + self.m_inv[(0, 1)] * y)
            + (self.m_inv[(0, 2)] * z + self.m_inv[(0, 3)]);
        let yp: Float = (self.m_inv[(1, 0)] * x + self.m_inv[(1, 1)] * y)
            + (self.m_inv[(1, 2)] * z + self.m_inv[(1, 3)]);
        let zp: Float = (self.m_inv[(2, 0)] * x + self.m_inv[(2, 1)] * y)
            + (self.m_inv[(2, 2)] * z + self.m_inv[(2, 3)]);
        let wp: Float = (self.m_inv[(3, 0)] * x + self.m_inv[(3, 1)] * y)
            + (self.m_inv[(3, 2)] * z + self.m_inv[(3, 3)]);

        let p_out_error = if rhs.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(self.m_inv[(0, 0)] * x)
                    + Float::abs(self.m_inv[(0, 1)] * y)
                    + Float::abs(self.m_inv[(0, 2)] * z));
            let y_err = gamma(3)
                * (Float::abs(self.m_inv[(1, 0)] * x)
                    + Float::abs(self.m_inv[(1, 1)] * y)
                    + Float::abs(self.m_inv[(1, 2)] * z));
            let z_err = gamma(3)
                * (Float::abs(self.m_inv[(2, 0)] * x)
                    + Float::abs(self.m_inv[(2, 1)] * y)
                    + Float::abs(self.m_inv[(2, 2)] * z));
            Vec3f::new(x_err, y_err, z_err)
        } else {
            let p_in_err = rhs.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[(0, 0)]) * p_in_err.x
                    + Float::abs(self.m_inv[(0, 1)]) * p_in_err.y
                    + Float::abs(self.m_inv[(0, 2)]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[(0, 0)] * x)
                        + Float::abs(self.m_inv[(0, 1)] * y)
                        + Float::abs(self.m_inv[(0, 2)] * z)
                        + Float::abs(self.m_inv[(0, 3)]));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[(1, 0)]) * p_in_err.x
                    + Float::abs(self.m_inv[(1, 1)]) * p_in_err.y
                    + Float::abs(self.m_inv[(1, 2)]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[(1, 0)] * x)
                        + Float::abs(self.m_inv[(1, 1)] * y)
                        + Float::abs(self.m_inv[(1, 2)] * z)
                        + Float::abs(self.m_inv[(1, 3)]));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[(2, 0)]) * p_in_err.x
                    + Float::abs(self.m_inv[(2, 1)]) * p_in_err.y
                    + Float::abs(self.m_inv[(2, 2)]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[(2, 0)] * x)
                        + Float::abs(self.m_inv[(2, 1)] * y)
                        + Float::abs(self.m_inv[(2, 2)] * z)
                        + Float::abs(self.m_inv[(2, 3)]));
            Vec3f::new(x_err, y_err, z_err)
        };

        if wp == 1.0 {
            Point3fi::from_errors(Point3f::new(xp, yp, zp), p_out_error.into())
        } else {
            Point3fi::from_errors(Point3f::new(xp, yp, zp), p_out_error.into()) / Interval::from(wp)
        }
    }
}

impl ApplyRayInverseTransform<Ray> for Transform {
    fn apply_ray_inverse(&self, ray: &Ray, t_max: Option<&mut Float>) -> Ray {
        let o: Point3fi = self.apply_inverse(Point3fi::from(ray.origin));
        let d: Vec3f = self.apply_inverse(ray.direction);
        let length_squared = d.length_squared();
        let o = if length_squared > 0.0 {
            let o_error = Vec3f::new(
                o.x.width() / 2.0,
                o.y.width() / 2.0,
                o.z.width() / 2.0,
            );
            let dt = d.abs().dot(o_error) / length_squared;
            if let Some(t_max) = t_max {
                *t_max -= dt;
            }
            o + Vec3fi::from(d * dt)
        } else {
            o
        };
        Ray::new_with_medium_time(Point3f::from(o), d, ray.time, ray.medium.clone())
    }
}

impl ApplyRayInverseTransform<RayDifferential> for Transform {
    fn apply_ray_inverse(&self, ray: &RayDifferential, t_max: Option<&mut Float>) -> RayDifferential {
        let tr = self.apply_ray_inverse(&ray.ray, t_max);

        let aux: Option<AuxiliaryRays> = if let Some(aux) = &ray.aux {
            let rx_origin = self.apply_inverse(aux.rx_origin);
            let rx_direction = self.apply_inverse(aux.rx_direction);
            let ry_origin = self.apply_inverse(aux.ry_origin);
            let ry_direction = self.apply_inverse(aux.ry_direction);
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

impl ApplyTransform<Transform> for Transform {
    fn apply(&self, rhs: Transform) -> Transform {
        Transform {
            m: self.m * rhs.m,
            m_inv: rhs.m_inv * self.m_inv,
        }
    }
}

fn apply_point(m: &Mat4, p: &Point3f) -> Point3f {
    let xp = m[(0, 0)] * p.x + m[(0, 1)] * p.y + m[(0, 2)] * p.z + m[(0, 3)];
    let yp = m[(1, 0)] * p.x + m[(1, 1)] * p.y + m[(1, 2)] * p.z + m[(1, 3)];
    let zp = m[(2, 0)] * p.x + m[(2, 1)] * p.y + m[(2, 2)] * p.z + m[(2, 3)];
    let wp = m[(3, 0)] * p.x + m[(3, 1)] * p.y + m[(3, 2)] * p.z + m[(3, 3)];
    if wp == 1.0 {
        Point3f::new(xp, yp, zp)
    } else {
        Point3f::new(xp, yp, zp) / wp
    }
}

fn apply_vector(m: &Mat4, v: &Vec3f) -> Vec3f {
    Vec3f::new(
        m[(0, 0)] * v.x + m[(0, 1)] * v.y + m[(0, 2)] * v.z,
        m[(1, 0)] * v.x + m[(1, 1)] * v.y + m[(1, 2)] * v.z,
        m[(2, 0)] * v.x + m[(2, 1)] * v.y + m[(2, 2)] * v.z,
    )
}

fn apply_normal(m: &Mat4, n: &Normal3f) -> Normal3f {
    Normal3f::new(
        m[(0, 0)] * n.x + m[(1, 0)] * n.y + m[(2, 0)] * n.z,
        m[(0, 1)] * n.x + m[(1, 1)] * n.y + m[(2, 1)] * n.z,
        m[(0, 2)] * n.x + m[(1, 2)] * n.y + m[(2, 2)] * n.z,
    )
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
        let scaled = scale.apply(p);
        assert_eq!(Normal3f::new(0.5, 0.6666667, 0.75), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);

        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::from_scale(Vec3f::new(2.0, 2.0, 2.0));
        let scaled = scale.apply(p);
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn bb_transform() {
        let b = Bounds3f::from_points(vec![Point3f::ZERO, Point3f::ONE]);
        let m = Transform::from_translation(Point3f::ONE);
        assert_eq!(Bounds3f::from_points(vec![Point3f::ONE, Point3f::ONE * 2.0]), m.apply(b));
    }

    #[test]
    fn rotate_from_to() {
        let from = Vec3f::new(2.0, 4.0, 1.0).normalize();
        let to = Vec3f::new(3.0, 1.0, 4.0).normalize();
        let r = Transform::from_rotation_delta(from, to);
        assert_abs_diff_eq!(to, r.apply(from), epsilon = 2e-6);

        let from = Vec3f::new(-1.0, -5.0, 3.0).normalize();
        let to = Vec3f::new(3.0, 1.0, -2.0).normalize();
        let r = Transform::from_rotation_delta(from, to);
        assert_abs_diff_eq!(to, r.apply(from), epsilon = 2e-6);
    }
}
