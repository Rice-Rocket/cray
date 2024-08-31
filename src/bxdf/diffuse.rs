use crate::{abs_cos_theta, color::sampled::SampledSpectrum, same_hemisphere, sampling::{cosine_hemisphere_pdf, sample_cosine_hemisphere}, Float, Point2f, Vec3f, FRAC_1_PI};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct DiffuseBxDF {
    r: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(r: SampledSpectrum) -> DiffuseBxDF {
        DiffuseBxDF { r }
    }
}

impl AbstractBxDF for DiffuseBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::from_const(0.0);
        }

        self.r * FRAC_1_PI
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

        let pdf = cosine_hemisphere_pdf(abs_cos_theta(wi));

        Some(BSDFSample::new(
            self.r * FRAC_1_PI,
            wi,
            pdf,
            BxDFFlags::DIFFUSE_REFLECTION,
        ))
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 || !same_hemisphere(wo, wi) {
            0.0
        } else {
            cosine_hemisphere_pdf(abs_cos_theta(wi))
        }
    }

    fn flags(&self) -> BxDFFlags {
        if self.r.is_zero() {
            BxDFFlags::UNSET
        } else {
            BxDFFlags::DIFFUSE_REFLECTION
        }
    }

    fn regularize(&mut self) {
        // Do nothing
    }
}
