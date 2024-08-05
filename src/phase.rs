use crate::{scattering::{henyey_greenstein, sample_henyey_greenstein}, Dot, Float, Point2f, Vec3f};

pub trait AbstractPhaseFunction {
    fn p(&self, wo: Vec3f, wi: Vec3f) -> Float;

    fn sample_p(&self, wo: Vec3f, u: Point2f) -> Option<PhaseFunctionSample>;

    fn pdf(&self, wo: Vec3f, wi: Vec3f) -> Float;
}

#[derive(Debug, Clone, Copy)]
pub enum PhaseFunction {
    HenyeyGreenstein(HGPhaseFunction),
}

impl PhaseFunction {
    pub fn create() -> PhaseFunction {
        todo!()
    }
}

impl AbstractPhaseFunction for PhaseFunction {
    fn p(&self, wo: Vec3f, wi: Vec3f) -> Float {
        match self {
            PhaseFunction::HenyeyGreenstein(f) => f.p(wo, wi),
        }
    }

    fn sample_p(&self, wo: Vec3f, u: Point2f) -> Option<PhaseFunctionSample> {
        match self {
            PhaseFunction::HenyeyGreenstein(f) => f.sample_p(wo, u),
        }
    }

    fn pdf(&self, wo: Vec3f, wi: Vec3f) -> Float {
        match self {
            PhaseFunction::HenyeyGreenstein(f) => f.pdf(wo, wi),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HGPhaseFunction {
    g: Float,
}

impl HGPhaseFunction {
    pub fn create() -> HGPhaseFunction {
        todo!()
    }

    pub fn new(g: Float) -> HGPhaseFunction {
        HGPhaseFunction { g }
    }
}

impl AbstractPhaseFunction for HGPhaseFunction {
    fn p(&self, wo: Vec3f, wi: Vec3f) -> Float {
        henyey_greenstein(wo.dot(wi), self.g)
    }

    fn sample_p(&self, wo: Vec3f, u: Point2f) -> Option<PhaseFunctionSample> {
        let (pdf, wi) = sample_henyey_greenstein(wo, self.g, u);
        Some(PhaseFunctionSample { p: pdf, wi, pdf })
    }

    fn pdf(&self, wo: Vec3f, wi: Vec3f) -> Float {
        self.p(wo, wi)
    }
}

pub struct PhaseFunctionSample {
    pub p: Float,
    pub wi: Vec3f,
    pub pdf: Float,
}
