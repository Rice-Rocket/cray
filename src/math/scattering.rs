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

pub fn fresnel_moment_1(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;

    if eta < 1.0 {
        0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3
            + 2.49277 * eta4 - 0.68441 * eta5
    } else {
        -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3
            - 1.27198 * eta4 + 0.12746 * eta5
    }
}

pub fn fresnel_moment_2(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;

    if eta < 1.0 {
        0.27614 - 0.87350 * eta + 1.12077 * eta2 - 0.65095 * eta3
            + 0.07883 * eta4 + 0.04860 * eta5
    } else {
        let r_eta = 1.0 / eta;
        let r_eta2 = r_eta * r_eta; 
        let r_eta3 = r_eta2 * r_eta;
        -547.033 + 45.3087 * r_eta3 - 218.725 * r_eta2 + 458.843 * r_eta
            + 404.557 * eta - 189.519 * eta2 + 54.9327 * eta3 - 9.00603 * eta4
            + 0.63942 * eta5
    }
}

pub fn fr_dielectric(mut cos_theta_i: Float, mut eta: Float) -> Float {
    cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }
    
    let sin2_theta_i = 1.0 - sqr(cos_theta_i);
    let sin2_theta_t = sin2_theta_i / sqr(eta);
    if sin2_theta_t >= 1.0 {
        return 1.0;
    }
    let cos_theta_t = safe::sqrt(1.0 - sin2_theta_t);

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
    (sqr(r_parl) + sqr(r_perp)) / 2.0
}
