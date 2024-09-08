use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{abs_cos_theta, color::sampled::SampledSpectrum, lerp, phase::{AbstractPhaseFunction, HGPhaseFunction}, same_hemisphere, sampling::{power_heuristic, sample_exponential}, Float, Point2f, Vec3f, ONE_MINUS_EPSILON, PI};

use super::{conductor::ConductorBxDF, dielectric::DielectricBxDF, diffuse::DiffuseBxDF, AbstractBxDF, BSDFSample, BxDFFlags, BxDFReflTransFlags, TransportMode};

#[derive(Debug, Clone)]
pub struct CoatedDiffuseBxDF {
    bxdf: LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>,
}

impl CoatedDiffuseBxDF {
    pub fn new(
        top: DielectricBxDF,
        bottom: DiffuseBxDF,
        thickness: Float,
        albedo: SampledSpectrum,
        g: Float,
        max_depth: i32,
        n_samples: i32
    ) -> CoatedDiffuseBxDF {
        CoatedDiffuseBxDF {
            bxdf: LayeredBxDF::new(top, bottom, thickness, albedo, g, max_depth, n_samples)
        }
    }
}

impl AbstractBxDF for CoatedDiffuseBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        self.bxdf.f(wo, wi, mode)
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        self.bxdf.sample_f(wo, uc, u, mode, sample_flags)
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        self.bxdf.pdf(wo, wi, mode, sample_flags)
    }

    fn flags(&self) -> BxDFFlags {
        self.bxdf.flags()
    }

    fn regularize(&mut self) {
        self.bxdf.regularize()
    }
}

#[derive(Debug, Clone)]
pub struct CoatedConductorBxDF {
    bxdf: LayeredBxDF<DielectricBxDF, ConductorBxDF, true>,
}

impl CoatedConductorBxDF {
    pub fn new(
        top: DielectricBxDF,
        bottom: ConductorBxDF,
        thickness: Float,
        albedo: SampledSpectrum,
        g: Float,
        max_depth: i32,
        n_samples: i32
    ) -> CoatedConductorBxDF {
        CoatedConductorBxDF {
            bxdf: LayeredBxDF::new(top, bottom, thickness, albedo, g, max_depth, n_samples)
        }
    }
}

impl AbstractBxDF for CoatedConductorBxDF {
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        self.bxdf.f(wo, wi, mode)
    }

    fn sample_f(
        &self,
        wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        self.bxdf.sample_f(wo, uc, u, mode, sample_flags)
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        self.bxdf.pdf(wo, wi, mode, sample_flags)
    }

    fn flags(&self) -> BxDFFlags {
        self.bxdf.flags()
    }

    fn regularize(&mut self) {
        self.bxdf.regularize()
    }
}

#[derive(Debug, Clone)]
struct LayeredBxDF<TopBxDF, BottomBxDF, const TWO_SIDED: bool>
where
    TopBxDF: AbstractBxDF,
    BottomBxDF: AbstractBxDF,
{
    top: TopBxDF,
    bottom: BottomBxDF,
    thickness: Float,
    g: Float,
    albedo: SampledSpectrum,
    max_depth: i32,
    n_samples: i32,
}

impl<TopBxDF, BottomBxDF, const TWO_SIDED: bool> LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED>
where
    TopBxDF: AbstractBxDF,
    BottomBxDF: AbstractBxDF,
{
    pub fn new(
        top: TopBxDF,
        bottom: BottomBxDF,
        thickness: Float,
        albedo: SampledSpectrum,
        g: Float,
        max_depth: i32,
        n_samples: i32,
    ) -> LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED> {
        LayeredBxDF {
            top,
            bottom,
            thickness,
            g,
            albedo,
            max_depth,
            n_samples,
        }
    }

    fn tr(&self, dz: Float, w: Vec3f) -> Float {
        if Float::abs(dz) <= Float::MIN {
            1.0
        } else {
            // TODO: use fast_exp()
            Float::exp(-Float::abs(dz / w.z))
        }
    }
}

impl<TopBxDF, BottomBxDF, const TWO_SIDED: bool> AbstractBxDF for LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED>
where
    TopBxDF: AbstractBxDF,
    BottomBxDF: AbstractBxDF,
{
    fn f(&self, mut wo: Vec3f, mut wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        let mut f = SampledSpectrum::from_const(0.0);

        if TWO_SIDED && wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }

        let entered_top = TWO_SIDED || wo.z > 0.0;
        let enter_interface = if entered_top {
            TopOrBottomBxDF {
                top: Some(&self.top),
                bottom: None,
            }
        } else {
            TopOrBottomBxDF {
                top: None,
                bottom: Some(&self.bottom),
            }
        };

        let (exit_interface, non_exit_interface) = if same_hemisphere(wo, wi) ^ entered_top {
            (
                TopOrBottomBxDF {
                    top: None,
                    bottom: Some(&self.bottom),
                },
                TopOrBottomBxDF {
                    top: Some(&self.top),
                    bottom: None,
                }
            )
        } else {
            (
                TopOrBottomBxDF {
                    top: Some(&self.top),
                    bottom: None,
                },
                TopOrBottomBxDF {
                    top: None,
                    bottom: Some(&self.bottom),
                }
            )
        };

        let exit_z = if same_hemisphere(wo, wi) ^ entered_top {
            0.0
        } else {
            self.thickness
        };

        if same_hemisphere(wo, wi) {
            f = enter_interface.f(wo, wi, mode) * self.n_samples as Float;
        }

        // TODO: use seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = || -> Float {
            let v: Float = rng.gen();
            Float::min(v, ONE_MINUS_EPSILON)
        };

        for _s in 0..self.n_samples {
            let uc = r();
            let Some(wos) = enter_interface.sample_f(
                wo, uc, Point2f::new(r(), r()), mode, BxDFReflTransFlags::TRANSMISSION
            ) else { continue };

            if wos.f.is_zero() || wos.pdf == 0.0 || wos.wi.z == 0.0 {
                continue;
            }

            let uc = r();
            let wis_mode = match mode {
                TransportMode::Radiance => TransportMode::Importance,
                TransportMode::Importance => TransportMode::Radiance,
            };

            let Some(wis) = exit_interface.sample_f(
                wi, uc, Point2f::new(r(), r()), wis_mode, BxDFReflTransFlags::TRANSMISSION
            ) else { continue };

            if wis.f.is_zero() || wis.pdf == 0.0 || wis.wi.z == 0.0 {
                continue;
            }

            let mut beta = wos.f * abs_cos_theta(wos.wi) / wos.pdf;
            let mut z = if entered_top {
                self.thickness
            } else {
                0.0
            };

            let mut w = wos.wi;
            let phase = HGPhaseFunction::new(self.g);

            for depth in 0..self.max_depth {
                if depth > 3 && beta.max_component_value() < 0.25 {
                    let q = Float::max(0.0, 1.0 - beta.max_component_value());
                    if r() < q {
                        break;
                    }
                    beta /= 1.0 - q;
                }

                if self.albedo.is_zero() {
                    z = if z == self.thickness {
                        0.0
                    } else {
                        self.thickness
                    };
                    beta = beta * self.tr(self.thickness, w);
                } else {
                    let sigma_t = 1.0;
                    let dz = sample_exponential(r(), sigma_t / Float::abs(w.z));
                    let zp = if w.z > 0.0 {
                        z + dz
                    } else {
                        z - dz
                    };

                    if z == zp {
                        continue;
                    }

                    if 0.0 < zp && zp < self.thickness {
                        let wt =  if !exit_interface.flags().is_specular() {
                            power_heuristic(1, wis.pdf, 1, phase.pdf(-w, -wis.wi))
                        } else {
                            1.0
                        };
                        f += beta * self.albedo * phase.p(-w, -wis.wi) * wt * self.tr(zp - exit_z, wis.wi) * wis.f / wis.pdf;

                        let u = Point2f::new(r(), r());
                        let Some(ps) = phase.sample_p(-w, u) else { continue };

                        beta *= (self.albedo * ps.p / ps.pdf);
                        w = ps.wi;
                        z = zp;

                        if ((z < exit_z && w.z > 0.0) || (z > exit_z && w.z < 0.0))
                        && !exit_interface.flags().is_specular() {
                            let f_exit = exit_interface.f(-w, wi, mode);
                            if !f_exit.is_zero() {
                                let exit_pdf = exit_interface.pdf(-w, wi, mode, BxDFReflTransFlags::TRANSMISSION);
                                let wt = power_heuristic(1, ps.pdf, 1, exit_pdf);
                                f += beta * self.tr(zp - exit_z, ps.wi) * f_exit * wt;
                            }
                        }

                        continue;
                    }

                    z = Float::clamp(zp, 0.0, self.thickness);
                }

                if z == exit_z {
                    let uc = r();
                    let Some(bs) = exit_interface.sample_f(
                        -w, uc, Point2f::new(r(), r()), mode, BxDFReflTransFlags::REFLECTION,
                    ) else { break };

                    if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
                        break;
                    }

                    beta *= (bs.f * abs_cos_theta(bs.wi) / bs.pdf);
                    w = bs.wi;
                } else {
                    if !non_exit_interface.flags().is_specular() {
                        let wt = if !exit_interface.flags().is_specular() {
                            power_heuristic(1, wis.pdf, 1, non_exit_interface.pdf(-w, -wis.wi, mode, BxDFReflTransFlags::ALL))
                        } else {
                            1.0
                        };
                        f += beta * non_exit_interface.f(-w, -wis.wi, mode) * abs_cos_theta(wis.wi)
                            * wt * self.tr(self.thickness, wis.wi) * wis.f / wis.pdf;
                    }

                    let uc = r();
                    let u = Point2f::new(r(), r());
                    let Some(bs) = non_exit_interface.sample_f(
                        -w, uc, u, mode, BxDFReflTransFlags::REFLECTION
                    ) else { break };

                    if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
                        break;
                    }

                    beta *= (bs.f * abs_cos_theta(bs.wi) / bs.pdf);
                    w = bs.wi;

                    if !exit_interface.flags().is_specular() {
                        let f_exit = exit_interface.f(-w, wi, mode);
                        if !f_exit.is_zero() {
                            let wt = if !non_exit_interface.flags().is_specular() {
                                let exit_pdf = exit_interface.pdf(-w, wi, mode, BxDFReflTransFlags::TRANSMISSION);
                                power_heuristic(1, bs.pdf, 1, exit_pdf)
                            } else {
                                1.0
                            };
                            f += beta * self.tr(self.thickness, bs.wi) * f_exit * wt;
                        }
                    }
                }
            }
        }

        f / self.n_samples as Float
    }

    fn sample_f(
        &self,
        mut wo: Vec3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        debug_assert!(sample_flags == BxDFReflTransFlags::ALL);

        let flip_wi = if TWO_SIDED && wo.z < 0.0 {
            wo = -wo;
            true
        } else {
            false
        };

        let entered_top = TWO_SIDED || wo.z > 0.0;
        let mut bs = if entered_top {
            self.top.sample_f(wo, uc, u, mode, BxDFReflTransFlags::ALL)?
        } else {
            self.bottom.sample_f(wo, uc, u, mode, BxDFReflTransFlags::ALL)?
        };

        if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
            return None;
        }

        if bs.is_reflection() {
            if flip_wi {
                bs.wi = -bs.wi;
            }

            bs.pdf_is_proportional = true;
            return Some(bs);
        }

        let mut w = bs.wi;
        let mut specular_path = bs.is_specular();

        // TODO: use seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = || -> Float {
            let v: Float = rng.gen();
            Float::min(v, ONE_MINUS_EPSILON)
        };

        let mut f = bs.f * abs_cos_theta(bs.wi);
        let mut pdf = bs.pdf;
        let mut z = if entered_top { self.thickness } else { 0.0 };
        let phase = HGPhaseFunction::new(self.g);

        for depth in 0..self.max_depth {
            let rr_beta = f.max_component_value() / pdf;
            if depth > 3 && rr_beta < 0.25 {
                let q = Float::max(0.0, 1.0 - rr_beta);
                if r() < q {
                    return None;
                }
                pdf *= 1.0 - q;
            }

            if w.z == 0.0 {
                return None;
            }

            if !self.albedo.is_zero() {
                let sigma_t = 1.0;
                let dz = sample_exponential(r(), sigma_t / abs_cos_theta(w));
                let zp = if w.z > 0.0 {
                    z + dz
                } else {
                    z - dz
                };

                if zp == z {
                    return None;
                }

                if 0.0 < zp && zp < self.thickness {
                    let ps = phase.sample_p(-w, Point2f::new(r(), r()))?;
                    if ps.pdf == 0.0 || ps.wi.z == 0.0 {
                        return None;
                    }

                    f *= self.albedo * ps.p;
                    pdf *= ps.pdf;
                    specular_path = false;
                    w = ps.wi;
                    z = zp;
                    
                    continue;
                }

                z = Float::clamp(zp, 0.0, self.thickness);
                if z == 0.0 {
                    debug_assert!(w.z < 0.0);
                } else {
                    debug_assert!(w.z > 0.0);
                }
            } else {
                z = if z == self.thickness { 0.0 } else { self.thickness };
                f = f * self.tr(self.thickness, w);
            }

            let interface = if z == 0.0 {
                TopOrBottomBxDF {
                    top: None,
                    bottom: Some(&self.bottom),
                }
            } else {
                TopOrBottomBxDF {
                    top: Some(&self.top),
                    bottom: None,
                }
            };

            let uc = r();
            let u = Point2f::new(r(), r());
            let bs = interface.sample_f(-w, uc, u, mode, BxDFReflTransFlags::ALL)?;
            if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
                return None;
            }

            f *= bs.f;
            pdf *= bs.pdf;
            specular_path &= bs.is_specular();
            w = bs.wi;

            if bs.is_transmission() {
                let mut flags = if same_hemisphere(wo, w) {
                    BxDFFlags::REFLECTION
                } else {
                    BxDFFlags::TRANSMISSION
                };

                flags |= if specular_path { BxDFFlags::SPECULAR } else { BxDFFlags::GLOSSY };

                if flip_wi {
                    w = -w;
                }

                return Some(BSDFSample {
                    f,
                    wi: w,
                    pdf,
                    flags,
                    eta: 1.0,
                    pdf_is_proportional: true,
                });
            }

            f = f * abs_cos_theta(bs.wi);
        }

        None
    }

    fn pdf(
        &self,
        mut wo: Vec3f,
        mut wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        debug_assert!(sample_flags == BxDFReflTransFlags::ALL);

        if TWO_SIDED && wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }

        // TODO: use seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = || -> Float {
            let v: Float = rng.gen();
            Float::min(v, ONE_MINUS_EPSILON)
        };

        let entered_top = TWO_SIDED || wo.z > 0.0;
        let mut pdf_sum = 0.0;

        if same_hemisphere(wo, wi) {
            let refl_flag = BxDFReflTransFlags::REFLECTION;
            pdf_sum += if entered_top {
                self.n_samples as Float * self.top.pdf(wo, wi, mode, refl_flag)
            } else {
                self.n_samples as Float * self.bottom.pdf(wo, wi, mode, refl_flag)
            };
        }

        for _s in 0..self.n_samples {
            if same_hemisphere(wo, wi) {
                let (r_interface, t_interface) = if entered_top {
                    (
                        TopOrBottomBxDF {
                            top: None,
                            bottom: Some(&self.bottom),
                        },
                        TopOrBottomBxDF {
                            top: Some(&self.top),
                            bottom: None,
                        },
                    )
                } else {
                    (
                        TopOrBottomBxDF {
                            top: Some(&self.top),
                            bottom: None,
                        },
                        TopOrBottomBxDF {
                            top: None,
                            bottom: Some(&self.bottom),
                        },
                    )
                };

                let trans = BxDFReflTransFlags::TRANSMISSION;
                let wos = t_interface.sample_f(wo, r(), Point2f::new(r(), r()), mode, trans);
                let wis_mode = match mode {
                    TransportMode::Radiance => TransportMode::Importance,
                    TransportMode::Importance => TransportMode::Radiance,
                };

                let wis = t_interface.sample_f(wi, r(), Point2f::new(r(), r()), wis_mode, trans);

                if let (Some(wos), Some(wis)) = (wos, wis) {
                    if !wos.f.is_zero() && wos.pdf > 0.0 && !wis.f.is_zero() && wis.pdf > 0.0 {
                        if !t_interface.flags().is_non_specular() {
                            pdf_sum += r_interface.pdf(-wos.wi, -wis.wi, mode, BxDFReflTransFlags::ALL);
                        } else if let Some(rs) = r_interface.sample_f(-wos.wi, r(), Point2f::new(r(), r()), mode, BxDFReflTransFlags::ALL) {
                            if r_interface.flags().is_specular() {
                                pdf_sum += t_interface.pdf(-rs.wi, wi, mode, BxDFReflTransFlags::ALL);
                            } else {
                                let r_pdf = r_interface.pdf(-wos.wi, -wis.wi, mode, BxDFReflTransFlags::ALL);
                                let wt = power_heuristic(1, wis.pdf, 1, r_pdf);
                                pdf_sum += wt * r_pdf;

                                let t_pdf = t_interface.pdf(-rs.wi, wi, mode, BxDFReflTransFlags::ALL);
                                let wt = power_heuristic(1, rs.pdf, 1, t_pdf);
                                pdf_sum += wt * t_pdf;
                            }
                        }
                    }
                }
            } else {
                let (to_interface, ti_interface) = if entered_top {
                    (
                        TopOrBottomBxDF {
                            top: Some(&self.top),
                            bottom: None,
                        },
                        TopOrBottomBxDF {
                            top: None,
                            bottom: Some(&self.bottom),
                        }
                    )
                } else {
                    (
                        TopOrBottomBxDF {
                            top: None,
                            bottom: Some(&self.bottom),
                        },
                        TopOrBottomBxDF {
                            top: Some(&self.top),
                            bottom: None,
                        }
                    )
                };

                let uc = r();
                let u = Point2f::new(r(), r());
                let Some(wos) = to_interface.sample_f(
                    wo, uc, u, mode, BxDFReflTransFlags::ALL,
                ) else { continue };

                if wos.f.is_zero() || wos.pdf == 0.0 || wos.wi.z == 0.0 || wos.is_reflection() {
                    continue;
                }

                let uc = r();
                let u = Point2f::new(r(), r());
                let wis_mode = match mode {
                    TransportMode::Radiance => TransportMode::Importance,
                    TransportMode::Importance => TransportMode::Radiance,
                };

                let Some(wis) = ti_interface.sample_f(
                    wi, uc, u, wis_mode, BxDFReflTransFlags::ALL
                ) else { continue };

                if wis.f.is_zero() || wis.pdf == 0.0 || wis.wi.z == 0.0 || wis.is_reflection() {
                    continue;
                }

                if to_interface.flags().is_specular() {
                    pdf_sum += ti_interface.pdf(-wos.wi, wi, mode, BxDFReflTransFlags::ALL);
                } else if ti_interface.flags().is_specular() {
                    pdf_sum += to_interface.pdf(wo, -wis.wi, mode, BxDFReflTransFlags::ALL);
                } else {
                    pdf_sum += (to_interface.pdf(wo, -wis.wi, mode, BxDFReflTransFlags::ALL)
                        + ti_interface.pdf(-wos.wi, wi, mode, BxDFReflTransFlags::ALL)) / 2.0;
                }
            }
        }

        lerp(1.0 / (4.0 * PI), pdf_sum / self.n_samples as Float, 0.9)
    }

    fn flags(&self) -> BxDFFlags {
        let top_flags = self.top.flags();
        let bottom_flags = self.bottom.flags();

        debug_assert!(top_flags.is_transmissive() || bottom_flags.is_transmissive());

        let mut flags = BxDFFlags::REFLECTION;

        if top_flags.is_specular() {
            flags |= BxDFFlags::SPECULAR;
        }

        if top_flags.is_diffuse() || bottom_flags.is_diffuse() || !self.albedo.is_zero() {
            flags |= BxDFFlags::DIFFUSE;
        } else if top_flags.is_glossy() || bottom_flags.is_glossy() {
            flags |= BxDFFlags::GLOSSY;
        }

        if top_flags.is_transmissive() && bottom_flags.is_transmissive() {
            flags |= BxDFFlags::TRANSMISSION;
        }

        flags
    }

    fn regularize(&mut self) {
        self.top.regularize();
        self.bottom.regularize();
    }
}

struct TopOrBottomBxDF<'a, TopBxDF, BottomBxDF>
where
    TopBxDF: AbstractBxDF,
    BottomBxDF: AbstractBxDF,
{
    top: Option<&'a TopBxDF>,
    bottom: Option<&'a BottomBxDF>,
}

impl<'a, TopBxDF, BottomBxDF> TopOrBottomBxDF<'a, TopBxDF, BottomBxDF>
where
    TopBxDF: AbstractBxDF,
    BottomBxDF: AbstractBxDF,
{
    fn f(&self, wo: Vec3f, wi: Vec3f, mode: TransportMode) -> SampledSpectrum {
        if let Some(top) = &self.top {
            top.f(wo, wi, mode)
        } else if let Some(bottom) = &self.bottom {
            bottom.f(wo, wi, mode)
        } else {
            panic!("TopOrBottomBxDF has no BxDFs to evaluate");
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
        if let Some(top) = &self.top {
            top.sample_f(wo, uc, u, mode, sample_flags)
        } else if let Some(bottom) = &self.bottom {
            bottom.sample_f(wo, uc, u, mode, sample_flags)
        } else {
            panic!("TopOrBottomBxDF has no BxDFs to evaluate");
        }
    }

    fn pdf(
        &self,
        wo: Vec3f,
        wi: Vec3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if let Some(top) = &self.top {
            top.pdf(wo, wi, mode, sample_flags)
        } else if let Some(bottom) = &self.bottom {
            bottom.pdf(wo, wi, mode, sample_flags)
        } else {
            panic!("TopOrBottomBxDF has no BxDFs to evaluate");
        }
    }

    fn flags(&self) -> BxDFFlags {
        if let Some(top) = &self.top {
            top.flags()
        } else if let Some(bottom) = &self.bottom {
            bottom.flags()
        } else {
            panic!("TopOrBottomBxDF has no BxDFs to evaluate");
        }
    }
}
