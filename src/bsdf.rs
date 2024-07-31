use crate::{bxdf::{AbstractBxDF, BSDFSample, BxDF, BxDFFlags, BxDFReflTransFlags, TransportMode}, color::sampled::SampledSpectrum, Float, Frame, Normal3f, Point2f, Vec3f};

pub struct BSDF {
    bxdf: BxDF,
    shading_frame: Frame,
}

impl BSDF {
    pub fn new(ns: Normal3f, dpdus: Vec3f, bxdf: BxDF) -> BSDF {
        let shading_frame = Frame::from_xz(dpdus.normalize(), Vec3f::from(ns));
        BSDF { bxdf, shading_frame }
    }

    pub fn flags(&self) -> BxDFFlags {
        self.bxdf.flags()
    }

    pub fn render_to_local(&self, v: Vec3f) -> Vec3f {
        self.shading_frame.localize(v)
    }

    pub fn local_to_render(&self, v: Vec3f) -> Vec3f {
        self.shading_frame.globalize(v)
    }

    pub fn f(
        &self,
        wo_render: Vec3f,
        wi_render: Vec3f,
        mode: TransportMode,
    ) -> SampledSpectrum {
        let wi = self.render_to_local(wi_render);
        let wo = self.render_to_local(wo_render);

        if wo.z == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        self.bxdf.f(wo, wi, mode)
    }

    pub fn sample_f(
        &self,
        wo_render: Vec3f,
        u: Float,
        u2: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let wo = self.render_to_local(wo_render);
        if wo.z == 0.0 || !((self.bxdf.flags().bits() & sample_flags.bits()) != 0) {
            return None;
        }

        let mut bs = self.bxdf.sample_f(wo, u, u2, mode, sample_flags)?;
        if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
            return None;
        }

        debug_assert!(bs.pdf >= 0.0);

        bs.wi = self.local_to_render(bs.wi);
        Some(bs)
    }

    pub fn pdf(
        &self,
        wo_render: Vec3f,
        wi_render: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let wo = self.render_to_local(wo_render);
        let wi = self.render_to_local(wi_render);
        if wo.z == 0.0 {
            return 0.0;
        }

        self.bxdf.pdf(wo, wi, mode, sample_flags)
    }

    pub fn rho_hd(&self, wo_render: Vec3f, uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        let wo = self.render_to_local(wo_render);
        self.bxdf.rho_hd(wo, uc, u2)
    }

    pub fn rho_hh(&self, u1: &[Point2f], uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        self.bxdf.rho_hh(u1, uc, u2)
    }

    pub fn regularize(&mut self) {
        self.bxdf.regularize()
    }
}
