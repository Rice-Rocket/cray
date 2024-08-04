use crate::{safe, spherical_direction, sqr, Float, Frame, Point2f, Vec3f, FRAC_1_4PI, TAU};

pub fn henyey_greenstein(cos_theta: Float, g: Float) -> Float {
    let g = Float::clamp(g, -0.99, 0.99);
    let denom = 1.0 + sqr(g) + 2.0 * g * cos_theta;
    FRAC_1_4PI * (1.0 - sqr(g)) / (denom * safe::sqrt(denom))
}

/// Returns (pdf, wi)
pub fn sample_henyey_greenstein(wo: Vec3f, g: Float, u: Point2f) -> (Float, Vec3f) {
    let g = Float::clamp(g, -0.99, 0.99);

    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * u.x
    } else {
        -1.0 / (2.0 * g) * (1.0 + sqr(g) - sqr((1.0 - sqr(g)) / (1.0 + g - 2.0 * g * u.x)))
    };

    let sin_theta = safe::sqrt(1.0 - sqr(cos_theta));
    let phi = TAU * u.x;
    let w_frame = Frame::from_z(wo);;
    let wi = w_frame.from_local(spherical_direction(sin_theta, cos_theta, phi));

    let pdf = henyey_greenstein(cos_theta, g);
    (pdf, wi)
}
