// Pbrt 3.8 Spherical Geometry

use crate::math::*;

/// Computes the area of a triangle given by three vectors on the unit sphere
/// `(a, b, c)` that correspond to the spherical triangle's vertices.
// TODO: Test this
#[inline]
pub fn spherical_triangle_area(a: &Unit<Vec3>, b: &Unit<Vec3>, c: &Unit<Vec3>) -> Scalar {
    (2.0 * (b.cross(c).dot(a)).atan2(1.0 + a.dot(b) + a.dot(c) + b.dot(c))).abs()
}

/// Converts a `(theta, phi)` pair to a unit `(x, y, z)` vector.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
///
/// Uses the `sin` and `cos` of theta.
// TODO: Test this
#[inline]
pub fn spherical_direction(sin_theta: Scalar, cos_theta: Scalar, phi: Scalar) -> Unit<Vec3> {
    Unit::new_unchecked(Vec3::new(
        sin_theta.clamp(-1.0, 1.0) * phi.cos(),
        sin_theta.clamp(-1.0, 1.0) * phi.sin(),
        cos_theta.clamp(-1.0, 1.0),
    ))
}

/// Computes the angle `theta` of a unit vector `v` on the unit sphere.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn spherical_theta(v: Unit<Vec3>) -> Scalar {
    math::safe::acos(v.z)
}

/// Computes the angle `phi` of a vector `v` on the unit sphere.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
#[inline]
// TODO: Test this
pub fn spherical_phi(v: Vec3) -> Scalar {
    let p = v.y.atan2(v.x);
    if p < 0.0 {
        p + math::TAU
    } else {
        p
    }
}

/// Computes the cosine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn cos_theta(w: Unit<Vec3>) -> Scalar {
    w.z
}

/// Computes the squared cosine of the `theta` component of given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn cos_2_theta(w: Unit<Vec3>) -> Scalar {
    w.z * w.z
}

/// Computes the sine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn sin_theta(w: Unit<Vec3>) -> Scalar {
    sin_2_theta(w).sqrt()
}

/// Computes the squared sine of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn sin_2_theta(w: Unit<Vec3>) -> Scalar {
    (1.0 - cos_2_theta(w)).max(0.0)
}

/// Computes the tangent of the `theta` component of given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn tan_theta(w: Unit<Vec3>) -> Scalar {
    sin_theta(w) / cos_theta(w)
}

/// Computes the squared tangent of the `theta` component of given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn tan_2_theta(w: Unit<Vec3>) -> Scalar {
    sin_2_theta(w) / cos_2_theta(w)
}

/// Computes the cosine of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn cos_phi(w: Unit<Vec3>) -> Scalar {
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
// TODO: Test this
#[inline]
pub fn cos_2_phi(w: Unit<Vec3>) -> Scalar {
    math::sqr(cos_phi(w))
}

/// Computes the sine of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn sin_phi(w: Unit<Vec3>) -> Scalar {
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
// TODO: Test this
#[inline]
pub fn sin_2_phi(w: Unit<Vec3>) -> Scalar {
    math::sqr(sin_phi(w))
}

/// Computes the tangent of the `phi` component of the given unit vector `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn tan_phi(w: Unit<Vec3>) -> Scalar {
    sin_phi(w) / cos_phi(w)
}

/// Computes the squared tangent of the `phi` component of the given unit vector
/// `w`.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn tan_2_phi(w: Unit<Vec3>) -> Scalar {
    sin_2_phi(w) / cos_2_phi(w)
}

/// Computes the cosine of the angle `delta phi` between two unit vectors.
///
/// Note that `theta` is the apparent `pitch` of the direction while `phi` is
/// the apparent `yaw` of the direction.
// TODO: Test this
#[inline]
pub fn cos_d_phi(wa: Unit<Vec3>, wb: Unit<Vec3>) -> Scalar {
    let waxy = math::sqr(wa.x) + math::sqr(wa.y);
    let wbxy = math::sqr(wb.x) + math::sqr(wb.y);

    if waxy == 0.0 || wbxy == 0.0 {
        return 1.0;
    };

    ((wa.x * wb.x + wa.y * wb.y) / (waxy * wbxy).sqrt()).clamp(-1.0, 1.0)
}


// TODO: Test this
pub fn equal_area_square_to_sphere(p: Vec2) -> Vec3 {
    let u = 2.0 * p.x - 1.0;
    let v = 2.0 * p.y - 1.0;
    let up = u.abs();
    let vp = v.abs();

    let sd = 1.0 - (up + vp);
    let d = sd.abs();
    let r = 1.0 - d;

    let phi = if r == 0.0 { 1.0 } else { ((vp - up) / r + 1.0) * math::FRAC_PI_4 };
    let z = (1.0 - r * r) * sd.signum();

    let cos_phi = phi.cos() * u.signum();
    let sin_phi = phi.sin() * v.signum();

    Vec3::new(cos_phi * r * math::safe::sqrt(2.0 - r * r), sin_phi * r * math::safe::sqrt(2.0 - r * r), z)
}


// TODO: Test this
pub struct OctahedralVec3 {
    x: u16,
    y: u16,
}

impl OctahedralVec3 {
    /// Builds a new [`OctahedralVec3`] from a Vec3.
    pub fn new(d: Unit<Vec3>) -> Self {
        let v = d.into_inner() / (d.x.abs() + d.y.abs() + d.z.abs());
        if v.z >= 0.0 {
            Self { x: Self::encode(v.x), y: Self::encode(v.y) }
        } else {
            Self { x: Self::encode((1.0 - v.y.abs()) * v.x.signum()), y: Self::encode((1.0 - v.x.abs()) * v.y.signum()) }
        }
    }

    /// Converts this [`OctahedralVec3`] back into a Vec3.
    pub fn to_vec3(self) -> Unit<Vec3> {
        let mut v = Vec3::new(-1.0 + 2.0 * (self.x as f32 / 65535.0), -1.0 + 2.0 * (self.y as f32 / 65535.0), 0.0);
        v.z = 1.0 - (v.x.abs() + v.y.abs());

        if v.z < 0.0 {
            let xo = v.x;
            v.x = (1.0 - v.y.abs()) * xo.signum();
            v.y = (1.0 - xo.abs()) * v.y.signum();
        }

        Unit::new_normalize(v)
    }

    fn encode(f: Scalar) -> u16 {
        (((f + 1.0) / 2.0).clamp(0.0, 1.0) * 65535.0).round() as u16
    }
}
