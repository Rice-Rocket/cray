use num::Complex;

use crate::{abs_cos_theta, color::sampled::{SampledSpectrum, NUM_SPECTRUM_SAMPLES}, cos_2_theta, cos_phi, lerp, safe, sampling::sample_uniform_disk_polar, sin_phi, spherical_direction, sqr, tan_2_theta, Dot, Float, Frame, Normal3f, Point2f, Vec2f, Vec3f, FRAC_1_4PI, PI, TAU};

#[inline]
pub fn reflect(wo: Vec3f, n: Normal3f) -> Vec3f {
    -wo + 2.0 * wo.dot(n) * n
}

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

#[inline]
pub fn fresnel_complex(cos_theta_i: Float, eta: Complex<Float>) -> Float {
    let cos_theta_i = Float::clamp(cos_theta_i, 0.0, 1.0);

    let sin2_theta_i = 1.0 - sqr(cos_theta_i);
    let sin2_theta_t = sin2_theta_i / sqr(eta);
    let cos_theta_t = Complex::sqrt(1.0 - sin2_theta_t);

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (Complex::norm_sqr(&r_parl) + Complex::norm_sqr(&r_perp)) / 2.0
}

#[inline]
pub fn fresnel_complex_spectral(cos_theta_i: Float, eta: SampledSpectrum, k: SampledSpectrum) -> SampledSpectrum {
    let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES {
        s[i] = fresnel_complex(cos_theta_i, Complex::new(eta[i], k[i]));
    }
    SampledSpectrum::new(s)
}

#[derive(Debug, Clone, Default)]
pub struct TrowbridgeReitzDistribution {
    alpha_x: Float,
    alpha_y: Float,
}

impl TrowbridgeReitzDistribution {
    pub fn new(ax: Float, ay: Float) -> Self {
        let d = Self {
            alpha_x: ax,
            alpha_y: ay,
        };

        if !d.effectively_smooth() {
            let alpha_x = Float::max(d.alpha_x, 1e-4);
            let alpha_y = Float::max(d.alpha_y, 1e-4);
            Self { alpha_x, alpha_y }
        } else {
            d
        }
    }

    #[inline]
    pub fn effectively_smooth(&self) -> bool {
        self.alpha_x < 1e-3 && self.alpha_y < 1e-3
    }

    #[inline]
    pub fn d(&self, wm: Vec3f) -> Float {
        let tan2_theta = tan_2_theta(wm);
        if tan2_theta.is_infinite() {
            return 0.0;
        }

        let cos4_theta = sqr(cos_2_theta(wm));
        if cos4_theta < 1e-16 {
            return 0.0;
        }

        let e = tan2_theta * (sqr(cos_phi(wm) / self.alpha_x) + sqr(sin_phi(wm) / self.alpha_y));
        1.0 / (PI * self.alpha_x * self.alpha_y * cos4_theta * sqr(1.0 + e))
    }

    pub fn g1(&self, w: Vec3f) -> Float {
        1.0 / (1.0 + self.lambda(w))
    }

    pub fn lambda(&self, w: Vec3f) -> Float {
        let tan2_theta = tan_2_theta(w);
        if tan2_theta.is_infinite() {
            return 0.0;
        }

        let alpha2 = sqr(cos_phi(w) * self.alpha_x) + sqr(sin_phi(w) * self.alpha_y);
        (-1.0 + Float::sqrt(1.0 + alpha2 * tan2_theta)) / 2.0
    }
    
    pub fn g(&self, wo: Vec3f, wi: Vec3f) -> Float {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }

    pub fn d_w(&self, w: Vec3f, wm: Vec3f) -> Float {
        self.g1(w) / abs_cos_theta(w) * self.d(wm) * w.dot(wm).abs()
    }

    pub fn pdf(&self, w: Vec3f, wm: Vec3f) -> Float {
        self.d_w(w, wm)
    }

    pub fn sample_wm(&self, w: Vec3f, u: Point2f) -> Vec3f {
        let mut wh = Vec3f::new(self.alpha_x * w.x, self.alpha_y * w.y, w.z).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        let t1 = if wh.z < 0.99999 {
            Vec3f::new(0.0, 0.0, 1.0).cross(wh).normalize()
        } else {
            Vec3f::new(1.0, 0.0, 0.0)
        };

        let t2 = wh.cross(t1);

        let mut p = sample_uniform_disk_polar(u);

        let h = Float::sqrt(1.0 - sqr(p.x));
        p.y = lerp(h, p.y, (1.0 + wh.z) / 2.0);

        let pz = Float::sqrt(Float::max(0.0, 1.0 - p.length_squared()));
        let nh = p.x * t1 + p.y * t2 + pz * wh;
        Vec3f::new(
            self.alpha_x * nh.x,
            self.alpha_y * nh.y,
            Float::max(1e-6, nh.z),
        ).normalize()
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        Float::sqrt(roughness)
    }

    pub fn regularize(&mut self) {
        if self.alpha_x < 0.3 {
            self.alpha_x = Float::clamp(2.0 * self.alpha_x, 0.1, 0.3);
        }

        if self.alpha_y < 0.3 {
            self.alpha_y = Float::clamp(2.0 * self.alpha_y, 0.1, 0.3);
        }
    }
}
