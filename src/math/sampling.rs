use crate::math::*;

pub fn sample_uniform_disk_concentric(u: Point2f) -> Point2f {
    let u_offset = u * 2.0 - Vec2f::new(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::ZERO;
    }

    let (theta, r) = if u_offset.x.abs() > u_offset.y.abs() {
        (FRAC_PI_4 * (u_offset.y / u_offset.x), u_offset.x)
    } else {
        (FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y), u_offset.y)
    };

    Point2f::new(theta.cos(), theta.sin()) * r
}

pub fn sample_uniform_sphere(u: Point2f) -> Vec3f {
    let z = 1.0 - 2.0 * u.x;
    let r = safe::sqrt(1.0 - z * z);
    let phi = TAU * u.y;

    Vec3f::new(r * phi.cos(), r * phi.sin(), z)
}
