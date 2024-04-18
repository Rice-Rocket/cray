// Pbrt 3.8.4 Spherical Geometry: Bounding Directions

use crate::math::*;

// TODO: Test this
#[derive(Clone, Copy)]
pub struct DirectionCone {
    pub w: Bivec3,
    pub cos_theta: Scalar,
}

impl DirectionCone {
    /// Creates a new [`DirectionCone`] from the given direction `w` and the
    /// cosine of half the central angle of a cone.
    pub fn new(w: Bivec3, cos_theta: Scalar) -> Self {
        Self { w, cos_theta }
    }

    pub fn from_direction(w: Bivec3) -> Self {
        Self { w, cos_theta: 1.0 }
    }

    pub fn bound_subtended_directions(b: &Bounds3, p_unit: &Bivec3) -> Self {
        let p = p_unit.get();
        let (center, radius) = b.bounding_sphere();
        if (p - center).magnitude_squared() < math::sqr(radius) {
            return Self::entire_sphere();
        }
        let w = Bivec3::new_normalize(p - center);
        let sin2_theta_max = math::sqr(radius) / (p - center).magnitude_squared();
        let cos_theta_max = math::safe::sqrt(1.0 - sin2_theta_max);
        Self { w, cos_theta: cos_theta_max }
    }

    pub fn union(&self, rhs: &Self) -> Self {
        if self.is_empty() {
            return *self;
        };
        if rhs.is_empty() {
            return *rhs;
        }

        let theta_a = math::safe::acos(self.cos_theta);
        let theta_b = math::safe::acos(rhs.cos_theta);
        let theta_d = self.w.angle(&rhs.w);

        if (theta_d + theta_b).min(math::PI) <= theta_a {
            return *self;
        }
        if (theta_d + theta_a).min(math::PI) <= theta_b {
            return *rhs;
        }

        let theta_o = (theta_a + theta_b + theta_d) / 2.0;
        if theta_o >= math::PI {
            return Self::entire_sphere();
        }

        let theta_r = theta_o - theta_a;
        let wr = self.w.cross(&rhs.w);
        if wr.magnitude_squared() == 0.0 {
            return Self::entire_sphere();
        }

        let w = Mat4::new_rotation(wr * theta_r) * Vec4::new(self.w.x(), self.w.y(), self.w.z(), 1.0);
        Self { w: Bivec3::new(w.xyz()), cos_theta: theta_o.cos() }
    }

    pub fn is_empty(&self) -> bool {
        self.cos_theta == Scalar::INFINITY
    }

    pub fn inside(&self, w: &Bivec3) -> bool {
        !self.is_empty() && self.w.dot(w) >= self.cos_theta
    }

    pub fn entire_sphere() -> Self {
        Self { w: Bivec3::new(Vec3::new(0.0, 0.0, 1.0)), cos_theta: -1.0 }
    }
}
