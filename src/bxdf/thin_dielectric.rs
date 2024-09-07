use crate::{abs_cos_theta, color::sampled::SampledSpectrum, scattering::fresnel_dielectric, sqr, Float, Point2f, Vec3f};

use super::{AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct ThinDielectricBxDF {
    eta: Float,
}

impl ThinDielectricBxDF {
    pub fn new(eta: Float) -> ThinDielectricBxDF {
        ThinDielectricBxDF { eta }
    }
}

impl AbstractBxDF for ThinDielectricBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let mut r = fresnel_dielectric(abs_cos_theta(wo), self.eta);
        let mut t = 1.0 - r;

        if r < 1.0 {
            r += sqr(t) * r / (1.0 - sqr(r));
            t = 1.0 - r;
        }

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
            let wi = Vec3f::new(-wo.x, -wo.y, wo.z);
            let fr = SampledSpectrum::from_const(r / abs_cos_theta(wi));
            Some(BSDFSample::new(
                fr,
                wi,
                pr / (pr + pt),
                BxDFFlags::SPECULAR_REFLECTION,
            ))
        } else {
            let wi = -wo;
            let ft = SampledSpectrum::from_const(t / abs_cos_theta(wi));
            Some(BSDFSample::new(
                ft,
                wi,
                pt / (pr + pt),
                BxDFFlags::SPECULAR_TRANSMISSION,
            ))
        }
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        0.0
    }

    fn flags(&self) -> BxDFFlags {
        BxDFFlags::REFLECTION | BxDFFlags::TRANSMISSION | BxDFFlags::SPECULAR
    }

    fn regularize(&mut self) {
        // Do nothing
    }
}
