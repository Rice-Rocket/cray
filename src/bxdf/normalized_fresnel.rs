use crate::{abs_cos_theta, color::sampled::SampledSpectrum, cos_theta, same_hemisphere, sampling::sample_cosine_hemisphere, scattering::{fr_dielectric, fresnel_moment_1}, Float, Point2f, Vec3f, FRAC_1_PI, PI};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct NormalizedFresnelBxDF {
    eta: Float
}

impl NormalizedFresnelBxDF {
    pub fn new(eta: Float) -> NormalizedFresnelBxDF {
        NormalizedFresnelBxDF { eta }
    }
}

impl AbstractBxDF for NormalizedFresnelBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::from_const(0.0);
        }

        let c = 1.0 - 2.0 * fresnel_moment_1(1.0 / self.eta);
        let mut f = SampledSpectrum::from_const((1.0 - fr_dielectric(cos_theta(wi), self.eta)) / (c * PI));

        if mode == TransportMode::Radiance {
            f = f * (self.eta * self.eta);
        }

        f
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            return None;
        }

        let mut wi = sample_cosine_hemisphere(u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }

        Some(BSDFSample::new(self.f(wo, wi, mode), wi, self.pdf(wo, wi, mode, sample_flags), BxDFFlags::DIFFUSE_REFLECTION))
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            return 0.0;
        }

        if same_hemisphere(wo, wi) { abs_cos_theta(wi) * FRAC_1_PI } else { 0.0 }
    }

    fn flags(&self) -> BxDFFlags {
        BxDFFlags::DIFFUSE_REFLECTION
    }

    fn regularize(&mut self) {
        // Do nothing
    }
}
