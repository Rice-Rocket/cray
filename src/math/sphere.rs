// Pbrt 3.8 Spherical Geometry

use fast_polynomial::poly_array;
use vect::Dot;

use crate::math::*;

/// Computes the area of a triangle given by three vectors on the unit sphere
/// `(a, b, c)` that correspond to the spherical triangle's vertices.
#[inline]
pub fn spherical_triangle_area(a: Vec3f, b: Vec3f, c: Vec3f) -> Float {
    debug_assert!(a.is_normalized());
    debug_assert!(b.is_normalized());
    debug_assert!(c.is_normalized());

    Float::abs(2.0 * Float::atan2(a.dot(b.cross(c)), 1.0 + a.dot(b) + a.dot(c) + b.dot(c)))
}

/// Converts a `(theta, phi)` pair to a unit `(x, y, z)` vector.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
///
/// Uses the `sin` and `cos` of theta.
#[inline]
pub fn spherical_direction(sin_theta: Float, cos_theta: Float, phi: Float) -> Vec3f {
    Vec3f::new(
        sin_theta.clamp(-1.0, 1.0) * phi.cos(),
        sin_theta.clamp(-1.0, 1.0) * phi.sin(),
        cos_theta.clamp(-1.0, 1.0),
    )
}

/// Computes the angle `theta` of a unit vector `v` on the unit sphere.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn spherical_theta(v: Vec3f) -> Float {
    debug_assert!(v.is_normalized());
    safe::acos(v.z)
}

/// Computes the angle `phi` of a vector `v` on the unit sphere.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn spherical_phi(v: Vec3f) -> Float {
    debug_assert!(v.is_normalized());

    let p = v.y.atan2(v.x);
    if p < 0.0 {
        p + TAU
    } else {
        p
    }
}

/// Computes the cosine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn cos_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    w.z
}

/// Computes the squared cosine of the `theta` component of given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn cos_2_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    w.z * w.z
}

/// Computes the sine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn sin_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sin_2_theta(w).sqrt()
}

/// Computes the squared sine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn sin_2_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    (1.0 - cos_2_theta(w)).max(0.0)
}

/// Computes the tangent of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn tan_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sin_theta(w) / cos_theta(w)
}

/// Computes the squared tangent of the `theta` component of given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn tan_2_theta(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sin_2_theta(w) / cos_2_theta(w)
}

/// Computes the cosine of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn cos_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        (w.x / sin_theta).clamp(-1.0, 1.0)
    }
}

/// Computes the squared cosine of the `phi` component of the given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn cos_2_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sqr(cos_phi(w))
}

/// Computes the sine of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn sin_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        0.0
    } else {
        (w.y / sin_theta).clamp(-1.0, 1.0)
    }
}

/// Computes the squared sine of the `phi` component of the given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn sin_2_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sqr(sin_phi(w))
}

/// Computes the tangent of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn tan_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sin_phi(w) / cos_phi(w)
}

/// Computes the squared tangent of the `phi` component of the given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn tan_2_phi(w: Vec3f) -> Float {
    debug_assert!(w.is_normalized());
    sin_2_phi(w) / cos_2_phi(w)
}

/// Computes the cosine of the angle `delta phi` between two unit vectors.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
pub fn cos_d_phi(wa: Vec3f, wb: Vec3f) -> Float {
    debug_assert!(wa.is_normalized());
    debug_assert!(wb.is_normalized());

    let waxy = sqr(wa.x) + sqr(wa.y);
    let wbxy = sqr(wb.x) + sqr(wb.y);

    if waxy == 0.0 || wbxy == 0.0 {
        return 1.0;
    };

    ((wa.x * wb.x + wa.y * wb.y) / (waxy * wbxy).sqrt()).clamp(-1.0, 1.0)
}

#[inline]
pub fn abs_cos_theta(w: Vec3f) -> Float {
    Float::abs(w.z)
}

#[inline]
pub fn same_hemisphere(w: Vec3f, wp: Vec3f) -> bool {
    w.z * wp.z > 0.0
}


pub fn equal_area_square_to_sphere(p: Vec2f) -> Vec3f {
    let u = 2.0 * p.x - 1.0;
    let v = 2.0 * p.y - 1.0;
    let up = u.abs();
    let vp = v.abs();

    let sd = 1.0 - (up + vp);
    let d = sd.abs();
    let r = 1.0 - d;

    let phi = if r == 0.0 { 1.0 } else { ((vp - up) / r + 1.0) * FRAC_PI_4 };
    let z = (1.0 - r * r) * sd.signum();

    let cos_phi = phi.cos() * u.signum();
    let sin_phi = phi.sin() * v.signum();

    Vec3f::new(cos_phi * r * safe::sqrt(2.0 - r * r), sin_phi * r * safe::sqrt(2.0 - r * r), z)
}

pub fn equal_area_sphere_to_square(d: Vec3f) -> Point2f {
    debug_assert!(d.length_squared() > 0.999 && d.length_squared() < 1.001);

    let x = d.x.abs();
    let y = d.y.abs();
    let z = d.z.abs();

    let r = safe::sqrt(1.0 - z);

    let a = Float::max(x, y);
    let b = Float::min(x, y);
    let b = if a == 0.0 { 0.0 } else { b / a };

    const T1: Float = 0.406758566246788489601959989e-5;
    const T2: Float = 0.636226545274016134946890922156;
    const T3: Float = 0.61572017898280213493197203466e-2;
    const T4: Float = -0.247333733281268944196501420480;
    const T5: Float = 0.881770664775316294736387951347e-1;
    const T6: Float = 0.419038818029165735901852432784e-1;
    const T7: Float = -0.251390972343483509333252996350e-1;

    let mut phi = poly_array(b, &[T1, T2, T3, T4, T5, T6, T7]);

    if x < y {
        phi = 1.0 - phi;
    }

    let mut v = phi * r;
    let mut u = r - v;

    if d.z < 0.0 {
        std::mem::swap(&mut u, &mut v);
        u = 1.0 - u;
        v = 1.0 - v;
    }

    u = Float::copysign(u, d.x);
    v = Float::copysign(v, d.y);

    Point2f::new(
        0.5 * (u + 1.0),
        0.5 * (v + 1.0),
    )
}

pub fn spherical_quad_area(a: Point3f, b: Point3f, c: Point3f, d: Point3f) -> Float {
    let axb = a.cross(b);
    let bxc = b.cross(c);
    let cxd = c.cross(d);
    let dxa = d.cross(a);

    if axb.length_squared() == 0.0 || bxc.length_squared() == 0.0 || cxd.length_squared() == 0.0 || dxa.length_squared() == 0.0 {
        return 0.0;
    }

    let axb = axb.normalize();
    let bxc = bxc.normalize();
    let cxd = cxd.normalize();
    let dxa = dxa.normalize();

    let alpha = dxa.angle_between(-axb);
    let beta = axb.angle_between(-bxc);
    let gamma = bxc.angle_between(-cxd);
    let delta = cxd.angle_between(-dxa);

    Float::abs(alpha + beta + gamma + delta - 2.0 * PI)
}


#[derive(Default, Debug, Clone, Copy)]
pub struct OctahedralVec3 {
    x: u16,
    y: u16,
}

impl OctahedralVec3 {
    /// Builds a new [`OctahedralVec3`] from a normalized Vec3.
    pub fn new(d: Vec3f) -> Self {
        debug_assert!(d.is_normalized());

        let v = d / (d.x.abs() + d.y.abs() + d.z.abs());
        if v.z >= 0.0 {
            Self { x: Self::encode(v.x), y: Self::encode(v.y) }
        } else {
            Self { x: Self::encode((1.0 - v.y.abs()) * v.x.signum()), y: Self::encode((1.0 - v.x.abs()) * v.y.signum()) }
        }
    }

    /// Converts this [`OctahedralVec3`] back into a normalized Vec3.
    pub fn to_vec3(self) -> Vec3f {
        let mut v = Vec3f::new(-1.0 + 2.0 * (self.x as Float / 65535.0), -1.0 + 2.0 * (self.y as Float / 65535.0), 0.0);
        v.z = 1.0 - (v.x.abs() + v.y.abs());

        if v.z < 0.0 {
            let xo = v.x;
            v.x = (1.0 - v.y.abs()) * xo.signum();
            v.y = (1.0 - xo.abs()) * v.y.signum();
        }

        v
    }

    fn encode(f: Float) -> u16 {
        (((f + 1.0) / 2.0).clamp(0.0, 1.0) * 65535.0).round() as u16
    }
}


#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{equal_area_sphere_to_square, equal_area_square_to_sphere, sampling::sample_uniform_sphere, Dot, Point2f};

    #[test]
    fn test_sphere_sampling() {
        let mut rng = rand::thread_rng();
        
        for _ in 0..100 {
            let u = Point2f::new(rng.gen(), rng.gen());
            let v = sample_uniform_sphere(u);
            let c = equal_area_sphere_to_square(v);
            let vp = equal_area_square_to_sphere(c.into());

            assert!(vp.length() > 0.9999 && vp.length() < 1.0001);
            assert!(v.dot(vp) > 0.9999);
        }
    }
}
