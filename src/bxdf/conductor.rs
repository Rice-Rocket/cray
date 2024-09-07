use crate::{abs_cos_theta, color::sampled::SampledSpectrum, same_hemisphere, scattering::{fresnel_complex_spectral, reflect, TrowbridgeReitzDistribution}, Dot, Float, Normal3f, Point2f, Vec3f};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct ConductorBxDF {
    mf_distribution: TrowbridgeReitzDistribution,
    eta: SampledSpectrum,
    k: SampledSpectrum,
}

impl ConductorBxDF {
    pub fn new(
        mf_distribution: TrowbridgeReitzDistribution,
        eta: SampledSpectrum,
        k: SampledSpectrum,
    ) -> ConductorBxDF {
        ConductorBxDF {
            mf_distribution,
            eta,
            k,
        }
    }
}

impl AbstractBxDF for ConductorBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::from_const(0.0);
        }

        if self.mf_distribution.effectively_smooth() {
            return SampledSpectrum::from_const(0.0);
        }

        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let wm = wi + wo;
        if wm.length_squared() == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let wm = wm.normalize();

        let f = fresnel_complex_spectral(wo.dot(wm).abs(), self.eta, self.k);

        self.mf_distribution.d(wm) * f * self.mf_distribution.g(wo, wi)
            / (4.0 * cos_theta_o * cos_theta_i)
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if (sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits()) == 0 {
            return None;
        }

        if self.mf_distribution.effectively_smooth() {
            let wi = Vec3f::new(-wo.x, -wo.y, wo.z);
            let f = fresnel_complex_spectral(abs_cos_theta(wi), self.eta, self.k) / abs_cos_theta(wi);
            return Some(BSDFSample::new(f, wi, 1.0, BxDFFlags::SPECULAR_REFLECTION));
        }

        if wo.z == 0.0 {
            return None;
        }

        let wm = self.mf_distribution.sample_wm(wo, u);
        let wi = reflect(wo, wm.into());
        if !same_hemisphere(wo, wi) {
            return None;
        }

        let pdf = self.mf_distribution.pdf(wo, wm) / (4.0 * wo.dot(wm).abs());

        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return None;
        }

        let mut f = fresnel_complex_spectral(wo.dot(wm).abs(), self.eta, self.k);
        f = self.mf_distribution.d(wm) * f * self.mf_distribution.g(wo, wi)
            / (4.0 * cos_theta_o * cos_theta_i);

        Some(BSDFSample::new(f, wi, pdf, BxDFFlags::GLOSSY_REFLECTION))
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0
        || !same_hemisphere(wo, wi)
        || self.mf_distribution.effectively_smooth() {
            return 0.0
        }

        let wm = wo + wi;
        if wm.length_squared() == 0.0 {
            return 0.0;
        }

        let wm = wm.normalize().facing(Vec3f::new(0.0, 0.0, 1.0));
        self.mf_distribution.pdf(wo, wm) / (4.0 * wo.dot(wm).abs())
    }

    fn flags(&self) -> BxDFFlags {
        if self.mf_distribution.effectively_smooth() {
            BxDFFlags::SPECULAR_REFLECTION
        } else {
            BxDFFlags::GLOSSY_REFLECTION
        }
    }

    fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}
