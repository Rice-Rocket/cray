// Pbrt 3.6 Rays

use crate::math::*;


pub trait RayLike {
    fn at(&self, t: Scalar) -> Point3f;
}


// TODO: Test this
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Ray {
    pub origin: Point3f,
    pub direction: UnitVec3f,
}

impl Default for Ray {
    fn default() -> Self {
        Self { origin: Point3f::new(0.0, 0.0, 0.0), direction: UnitVec3f::new(0.0, 0.0, 1.0) }
    }
}

impl Ray {
    /// Create a new [`Ray`] from an `origin` and `direction`.
    #[inline]
    pub const fn new(origin: Point3f, direction: UnitVec3f) -> Self {
        Self { origin, direction }
    }

    /// Create a new [`Ray`] from an `origin` with a direction pointing along
    /// the positive Z axis.
    #[inline]
    pub const fn from_origin(origin: Point3f) -> Self {
        Self { origin, direction: UnitVec3f::new(0.0, 0.0, 1.0) }
    }

    /// Create a new [`Ray`] from a `direction` with an origin at `(0, 0, 0)`.
    #[inline]
    pub const fn from_direction(direction: UnitVec3f) -> Self {
        Self { origin: Point3f::new(0.0, 0.0, 0.0), direction }
    }

    /// Returns whether or not this [`Ray`] contains an infinite value.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.origin.x.is_infinite()
            || self.origin.y.is_infinite()
            || self.origin.z.is_infinite()
            || self.direction.x.is_infinite()
            || self.direction.y.is_infinite()
            || self.direction.z.is_infinite()
    }

    /// Returns whether or not this [`Ray`] contains a NaN.
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.origin.x.is_nan()
            || self.origin.y.is_nan()
            || self.origin.z.is_nan()
            || self.direction.x.is_nan()
            || self.direction.y.is_nan()
            || self.direction.z.is_nan()
    }

    /// Returns whether or not this [`Ray`] is entirely made up of finite
    /// values.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.origin.x.is_finite()
            && self.origin.y.is_finite()
            && self.origin.z.is_finite()
            && self.direction.x.is_finite()
            && self.direction.y.is_finite()
            && self.direction.z.is_finite()
    }
}


impl RayLike for Ray {
    /// Computes the point along this [`Ray`] at a given `t`.
    #[inline]
    fn at(&self, t: Scalar) -> Point3f {
        self.origin + self.direction * t
    }
}


pub struct RayDifferential {
    pub ray: Ray,
    pub aux: Option<AuxiliaryRays>,
}

impl RayDifferential {
    /// Creates a new [`RayDifferential`].
    #[inline]
    pub fn new(ray: Ray, aux: Option<AuxiliaryRays>) -> RayDifferential {
        RayDifferential { ray, aux }
    }

    /// Scales the auxiliary rays given an estimated spacing of `s`.
    pub fn scale_differentials(&mut self, s: Scalar) {
        if let Some(aux) = &mut self.aux {
            aux.rx_origin = self.ray.origin + (aux.rx_origin - self.ray.origin) * s;
            aux.ry_origin = self.ray.origin + (aux.ry_origin - self.ray.origin) * s;

            aux.rx_direction = self.ray.direction + (aux.rx_direction - self.ray.direction) * s;
            aux.ry_direction = self.ray.direction + (aux.ry_direction - self.ray.direction) * s;
        }
    }
}

impl RayLike for RayDifferential {
    /// Computes the point along this [`Ray`] at a given `t`.
    #[inline]
    fn at(&self, t: Scalar) -> Point3f {
        self.ray.at(t)
    }
}


pub struct AuxiliaryRays {
    pub rx_origin: Point3f,
    pub rx_direction: UnitVec3f,
    pub ry_origin: Point3f,
    pub ry_direction: UnitVec3f,
}

impl AuxiliaryRays {
    /// Creates a new [`AuxiliaryRays`].
    #[inline]
    pub fn new(rx_origin: Point3f, rx_direction: UnitVec3f, ry_origin: Point3f, ry_direction: UnitVec3f) -> AuxiliaryRays {
        AuxiliaryRays { rx_origin, rx_direction, ry_origin, ry_direction }
    }
}
