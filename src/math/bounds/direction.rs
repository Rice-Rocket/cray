// Pbrt 3.8.4 Spherical Geometry: Bounding Directions

use crate::math::*;

#[derive(Clone, Copy)]
pub struct DirectionCone {
    pub w: Vec3f,
    pub cos_theta: Scalar,
}

impl DirectionCone {
    /// Creates a new [`DirectionCone`] from the given direction `w` and the
    /// cosine of half the central angle of a cone.
    pub fn new(w: Vec3f, cos_theta: Scalar) -> Self {
        debug_assert!(w.is_normalized());
        Self { w, cos_theta }
    }

    pub fn from_direction(w: Vec3f) -> Self {
        debug_assert!(w.is_normalized());
        Self { w, cos_theta: 1.0 }
    }

    pub fn bound_subtended_directions(b: Bounds3f, p: Point3f) -> Self {
        let (center, radius) = b.bounding_sphere();
        if (p - center).length_squared() < sqr(radius) {
            return Self::entire_sphere();
        }
        let w = Vec3f::from((p - center).normalize());
        let sin2_theta_max = sqr(radius) / (p - center).length_squared();
        let cos_theta_max = safe::sqrt(1.0 - sin2_theta_max);
        Self { w, cos_theta: cos_theta_max }
    }

    pub fn union(self, rhs: Self) -> Self {
        if self.is_empty() {
            return self;
        };
        if rhs.is_empty() {
            return rhs;
        }

        let theta_a = safe::acos(self.cos_theta);
        let theta_b = safe::acos(rhs.cos_theta);
        let theta_d = self.w.angle(rhs.w);

        if (theta_d + theta_b).min(PI) <= theta_a {
            return self;
        }
        if (theta_d + theta_a).min(PI) <= theta_b {
            return rhs;
        }

        let theta_o = (theta_a + theta_b + theta_d) / 2.0;
        if theta_o >= PI {
            return Self::entire_sphere();
        }

        let theta_r = theta_o - theta_a;
        let wr = self.w.cross(rhs.w);
        if wr.length_squared() == 0.0 {
            return Self::entire_sphere();
        }

        let w = Mat4::from_rotation(theta_r.sin(), theta_r.sin(), wr) * Point4f::new(self.w.x, self.w.y, self.w.z, 1.0);
        Self { w: Vec3f::from(w.xyz()), cos_theta: theta_o.cos() }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_theta == Scalar::INFINITY
    }

    pub fn inside(&self, w: Vec3f) -> bool {
        debug_assert!(w.is_normalized());

        !self.is_empty() && self.w.dot(w) >= self.cos_theta
    }

    pub fn entire_sphere() -> Self {
        Self { w: Vec3f::new(0.0, 0.0, 1.0), cos_theta: -1.0 }
    }
}
