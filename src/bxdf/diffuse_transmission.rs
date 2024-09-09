use crate::{abs_cos_theta, color::sampled::SampledSpectrum, same_hemisphere, sampling::{cosine_hemisphere_pdf, sample_cosine_hemisphere}, Float, Point2f, Vec3f, FRAC_1_PI};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct DiffuseTransmissionBxDF {
    r: SampledSpectrum,
    t: SampledSpectrum,
}

impl DiffuseTransmissionBxDF {
    pub fn new(r: SampledSpectrum, t: SampledSpectrum) -> DiffuseTransmissionBxDF {
        DiffuseTransmissionBxDF { r, t }
    }
}

impl AbstractBxDF for DiffuseTransmissionBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if same_hemisphere(wo, wi) {
            self.r * FRAC_1_PI
        } else {
            self.t * FRAC_1_PI
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
        let mut pr = self.r.max_component_value();
        let mut pt = self.t.max_component_value();

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
            let mut wi = sample_cosine_hemisphere(u);
            if wo.z < 0.0 {
                wi.z = -wi.z;
            }
            let pdf = cosine_hemisphere_pdf(abs_cos_theta(wi)) * pr / (pr + pt);
            Some(BSDFSample::new(self.f(wo, wi, mode), wi, pdf, BxDFFlags::DIFFUSE_REFLECTION))
        } else {
            let mut wi = sample_cosine_hemisphere(u);
            if wo.z > 0.0 {
                wi.z = -wi.z;
            }
            let pdf = cosine_hemisphere_pdf(abs_cos_theta(wi)) * pt / (pr + pt);
            Some(BSDFSample::new(self.f(wo, wi, mode), wi, pdf, BxDFFlags::DIFFUSE_TRANSMISSION))
        }
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let mut pr = self.r.max_component_value();
        let mut pt = self.t.max_component_value();

        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            pr = 0.0;
        }
        if sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits() == 0 {
            pt = 0.0;
        }

        if pr == 0.0 && pt == 0.0 {
            return 0.0;
        }

        if same_hemisphere(wo, wi) {
            pr / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))
        } else {
            pt / (pr + pt) * cosine_hemisphere_pdf(abs_cos_theta(wi))
        }
    }

    fn flags(&self) -> BxDFFlags {
        let mut flags = BxDFFlags::DIFFUSE;

        if !self.r.is_zero() {
            flags |= BxDFFlags::REFLECTION;
        }

        if !self.t.is_zero() {
            flags |= BxDFFlags::TRANSMISSION;
        }

        flags
    }

    fn regularize(&mut self) {
        // Do nothing
    }
}
