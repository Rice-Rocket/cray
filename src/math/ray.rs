// Pbrt 3.6 Rays

use crate::{math::*, media::Medium};


pub trait RayLike {
    fn at(&self, t: Float) -> Point3f;
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Ray {
    pub origin: Point3f,
    pub direction: Vec3f,
    pub time: Float,
    pub medium: Option<Medium>,
}

impl Default for Ray {
    fn default() -> Self {
        Self { origin: Point3f::new(0.0, 0.0, 0.0), direction: Vec3f::new(0.0, 0.0, 1.0), time: 0.0, medium: None }
    }
}

impl Ray {
    /// Create a new [`Ray`] from an `origin` and `direction`.
    #[inline]
    pub fn new(origin: Point3f, direction: Vec3f) -> Self {
        debug_assert!(direction.is_normalized());
        Self { origin, direction, time: 0.0, medium: None }
    }

    #[inline]
    pub fn new_with_medium(origin: Point3f, direction: Vec3f, medium: Option<Medium>) -> Self {
        Self { origin, direction, time: 0.0, medium }
    }

    /// Create a new [`Ray`] from an `origin`, `direction` and `time`.
    #[inline]
    pub fn new_with_time(origin: Point3f, direction: Vec3f, time: Float) -> Self {
        debug_assert!(direction.is_normalized());
        Self { origin, direction, time, medium: None }
    }

    #[inline]
    pub fn new_with_medium_time(origin: Point3f, direction: Vec3f, time: Float, medium: Option<Medium>) -> Self {
        debug_assert!(direction.is_normalized());
        Self { origin, direction, time, medium }
    }

    /// Create a new [`Ray`] from an `origin` with a direction pointing along
    /// the positive Z axis.
    #[inline]
    pub const fn from_origin(origin: Point3f) -> Self {
        Self { origin, direction: Vec3f::new(0.0, 0.0, 1.0), time: 0.0, medium: None }
    }

    /// Create a new [`Ray`] from a `direction` with an origin at `(0, 0, 0)`.
    #[inline]
    pub fn from_direction(direction: Vec3f) -> Self {
        debug_assert!(direction.is_normalized());
        Self { origin: Point3f::new(0.0, 0.0, 0.0), direction, time: 0.0, medium: None }
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

    pub fn offset_ray_origin(pi: Point3fi, n: Normal3f, w: Vec3f) -> Point3f {
        let d = Point3f::from(n).abs().dot(pi.error());
        let mut offset = Vec3f::from(n) * d;

        if w.dot(n.into()) < 0.0 {
            offset = -offset;
        }
        let po = Point3f::from(pi) + offset;
        po.zip(offset.into()).map(|(p, o)| 
            if o > 0.0 { next_float_up(p) } else if o < 0.0 { next_float_down(p) } else { p })
    }

    pub fn spawn_ray(pi: Point3fi, n: Normal3f, time: Float, d: Vec3f) -> Ray {
        Ray::new_with_time(Ray::offset_ray_origin(pi, n, d), d, time)
    }

    pub fn spawn_ray_to(p_from: Point3fi, n: Normal3f, time: Float, p_to: Point3f) -> Ray {
        let d = p_to - Point3f::from(p_from);
        Self::spawn_ray(p_from, n, time, d.into())
    }

    pub fn spawn_ray_to_both_offset(p_from: Point3fi, n_from: Normal3f, time: Float, p_to: Point3fi, n_to: Normal3f) -> Ray {
        let pf = Self::offset_ray_origin(p_from, n_from, (Point3f::from(p_to) - Point3f::from(p_from)).into());
        let pt = Self::offset_ray_origin(p_to, n_to, (pf - Point3f::from(p_to)).into());

        Ray {
            origin: pf,
            direction: (pt - pf).into(),
            time,
            medium: None,
        }
    }
}


impl RayLike for Ray {
    /// Computes the point along this [`Ray`] at a given `t`.
    #[inline]
    fn at(&self, t: Float) -> Point3f {
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
    pub fn scale_differentials(&mut self, s: Float) {
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
    fn at(&self, t: Float) -> Point3f {
        self.ray.at(t)
    }
}


pub struct AuxiliaryRays {
    pub rx_origin: Point3f,
    pub rx_direction: Vec3f,
    pub ry_origin: Point3f,
    pub ry_direction: Vec3f,
}

impl AuxiliaryRays {
    /// Creates a new [`AuxiliaryRays`].
    #[inline]
    pub fn new(rx_origin: Point3f, rx_direction: Vec3f, ry_origin: Point3f, ry_direction: Vec3f) -> AuxiliaryRays {
        debug_assert!(rx_direction.is_normalized());
        debug_assert!(ry_direction.is_normalized());
        AuxiliaryRays { rx_origin, rx_direction, ry_origin, ry_direction }
    }
}
