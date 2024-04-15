use crate::math::Vec3;


#[derive(Debug, PartialEq, Clone, Copy, Default)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Create a new [`Ray`] from an `origin` and `direction`
    pub const fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Create a new [`Ray`] from an `origin` with a direction pointing along
    /// the positive Z axis
    pub const fn from_origin(origin: Vec3) -> Self {
        Self { origin, direction: Vec3::new(0.0, 0.0, 1.0) }
    }

    /// Create a new [`Ray`] from a `direction` with an origin at `(0, 0, 0)`
    pub const fn from_direction(direction: Vec3) -> Self {
        Self { origin: Vec3::new(0.0, 0.0, 0.0), direction }
    }
}
