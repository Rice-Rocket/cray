use crate::{abs_cos_theta, color::sampled::SampledSpectrum, cos_theta, same_hemisphere, scattering::{fresnel_dielectric, reflect, refract, TrowbridgeReitzDistribution}, sqr, Dot, Float, Normal3f, Point2f, Vec3f};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct DielectricBxDF {
    eta: Float,
    mf_distribution: TrowbridgeReitzDistribution,
}

impl DielectricBxDF {
    pub fn new(eta: Float, mf_distribution: TrowbridgeReitzDistribution) -> DielectricBxDF {
        DielectricBxDF {
            eta,
            mf_distribution,
        }
    }
}

impl AbstractBxDF for DielectricBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            return SampledSpectrum::from_const(0.0);
        }

        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let etap = if !reflect {
            if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            }
        } else {
            1.0
        };

        let wm = wi * etap + wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let wm = wm.normalize().facing(Vec3f::new(0.0, 0.0, 1.0));
        if wm.dot(wi) * cos_theta_i < 0.0 || wm.dot(wo) * cos_theta_o < 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let f = fresnel_dielectric(wo.dot(wm), self.eta);
        if reflect {
            SampledSpectrum::from_const(
                self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * f
                    / Float::abs(4.0 * cos_theta_i * cos_theta_o),
            )
        } else {
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap) * cos_theta_i * cos_theta_o;
            let mut ft = self.mf_distribution.d(wm)
                * (1.0 - f)
                * self.mf_distribution.g(wo, wi)
                * Float::abs(wi.dot(wm) * wo.dot(wm) / denom);
            
            if mode == TransportMode::Radiance {
                ft /= sqr(etap)
            }

            SampledSpectrum::from_const(ft)
        }
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            let r = fresnel_dielectric(cos_theta(wo), self.eta);
            let t = 1.0 - r;

            let mut pr = r;
            let mut pt = t;

            if (sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits()) == 0 {
                pr = 0.0;
            }

            if (sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits()) == 0 {
                pt = 0.0;
            }

            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                let wi = Vec3f::new(-wo.x, -wo.y, wo.z);
                let fr = SampledSpectrum::from_const(r / abs_cos_theta(wi));
                Some(BSDFSample::new(
                    fr,
                    wi,
                    pr / (pr + pt),
                    BxDFFlags::SPECULAR_REFLECTION,
                ))
            } else if let Some((wi, etap)) = refract(wo, Normal3f::new(0.0, 0.0, 1.0), self.eta) {
                let mut ft = SampledSpectrum::from_const(t / abs_cos_theta(wi));
                if mode == TransportMode::Radiance {
                    ft /= sqr(etap);
                }

                Some(BSDFSample::new_with_eta(
                    ft,
                    wi,
                    pt / (pr + pt),
                    BxDFFlags::SPECULAR_TRANSMISSION,
                    etap
                ))
            } else {
                None
            }
        } else {
            let wm = self.mf_distribution.sample_wm(wo, u);
            let r = fresnel_dielectric(wo.dot(wm), self.eta);
            let t = 1.0 - r;
            
            let mut pr = r;
            let mut pt = t;

            if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
                pr = 0.0;
            }

            if sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits() == 0 {
                pt = 0.0;
            }

            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                let wi = reflect(wo, wm.into());
                if !same_hemisphere(wo, wi) {
                    return None;
                }

                let pdf = self.mf_distribution.pdf(wo, wm) / (4.0 * wo.dot(wm).abs()) * pr / (pr + pt);
                debug_assert!(!pdf.is_nan());

                let f = SampledSpectrum::from_const(
                    self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * r
                        / (4.0 * cos_theta(wi) * cos_theta(wo)),
                );
                Some(BSDFSample::new(f, wi, pdf, BxDFFlags::GLOSSY_REFLECTION))
            } else if let Some((wi, etap)) = refract(wo, wm.into(), self.eta) {
                if same_hemisphere(wo, wi) || wi.z == 0.0 {
                    return None;
                }

                let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
                let dwm_dwi = wi.dot(wm).abs() / denom;
                let pdf = self.mf_distribution.pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
                debug_assert!(!pdf.is_nan());

                let mut ft = SampledSpectrum::from_const(
                    t * self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi)
                        * Float::abs(wi.dot(wm) * wo.dot(wm) / (cos_theta(wi) * cos_theta(wo) * denom))
                );

                if mode == TransportMode::Radiance {
                    ft /= sqr(etap);
                }

                Some(BSDFSample::new_with_eta(
                    ft,
                    wi,
                    pdf,
                    BxDFFlags::GLOSSY_TRANSMISSION,
                    etap
                ))
            } else {
                None
            }
        }
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            return 0.0;
        }

        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let etap = if !reflect {
            if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            }
        } else {
            1.0
        };

        let wm = wi * etap + wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return 0.0;
        }

        let wm = wm.normalize().facing(Vec3f::new(0.0, 0.0, 1.0));

        if wm.dot(wi) * cos_theta_i < 0.0 || wm.dot(wo) * cos_theta_o < 0.0 {
            return 0.0;
        }

        let r = fresnel_dielectric(wo.dot(wm), self.eta);
        let t = 1.0 - r;

        let mut pr = r;
        let mut pt = t;

        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            pr = 0.0;
        }

        if sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits() == 0 {
            pt = 0.0;
        }

        if pr == 0.0 && pt == 0.0 {
            return 0.0;
        }

        if reflect {
            self.mf_distribution.pdf(wo, wm) / (4.0 * wo.dot(wm).abs()) * pr / (pr + pt)
        } else {
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
            let dwm_dwi = wi.dot(wm).abs() / denom;
            self.mf_distribution.pdf(wo, wm) * dwm_dwi * pt / (pr + pt)
        }
    }

    fn flags(&self) -> BxDFFlags {
        let flags = if self.eta == 1.0 {
            BxDFFlags::TRANSMISSION
        } else {
            BxDFFlags::REFLECTION | BxDFFlags::TRANSMISSION
        };

        let mf = if self.mf_distribution.effectively_smooth() {
            BxDFFlags::SPECULAR
        } else {
            BxDFFlags::GLOSSY
        };

        flags | mf
    }

    fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}
