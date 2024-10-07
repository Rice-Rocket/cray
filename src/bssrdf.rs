use crate::{bsdf::BSDF, bxdf::{normalized_fresnel::NormalizedFresnelBxDF, BxDF}, catmull_rom_weights, color::sampled::{SampledSpectrum, NUM_SPECTRUM_SAMPLES}, integrate_catmull_rom, interaction::{Interaction, SurfaceInteraction, SurfaceInteractionShading}, invert_catmull_rom, safe, sample_catmull_rom_2d, sampling::sample_exponential, scattering::{fr_dielectric, fresnel_moment_1, fresnel_moment_2, henyey_greenstein}, sqr, Float, Frame, Normal3f, Point2f, Point3f, Point3fi, Vec3f, FRAC_1_4PI, PI};

pub trait AbstractBSSRDF {
    type BxDF;

    fn sample_sp(&self, u1: Float, u2: Point2f) -> Option<BSSRDFProbeSegment>;
    fn probe_intersection_to_sample(&self, si: &SubsurfaceInteraction) -> BSSRDFSample;
}

pub enum BSSRDF {
    Tabulated(TabulatedBSSRDF),
}

impl AbstractBSSRDF for BSSRDF {
    type BxDF = NormalizedFresnelBxDF;

    fn sample_sp(&self, u1: Float, u2: Point2f) -> Option<BSSRDFProbeSegment> {
        match self {
            BSSRDF::Tabulated(b) => b.sample_sp(u1, u2),
        }
    }

    fn probe_intersection_to_sample(&self, si: &SubsurfaceInteraction) -> BSSRDFSample {
        match self {
            BSSRDF::Tabulated(b) => b.probe_intersection_to_sample(si),
        }
    }
}

pub struct BSSRDFSample {
    pub sp: SampledSpectrum,
    pub pdf: SampledSpectrum,
    pub sw: BSDF,
    pub wo: Vec3f,
}

#[derive(Debug, Clone, Default)]
pub struct SubsurfaceInteraction {
    pub pi: Point3fi,
    pub n: Normal3f,
    pub ns: Normal3f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dpdus: Vec3f,
    pub dpdvs: Vec3f,
}

impl SubsurfaceInteraction {
    pub fn p(&self) -> Point3f {
        self.pi.into()
    }
}

impl From<&SurfaceInteraction> for SubsurfaceInteraction {
    fn from(value: &SurfaceInteraction) -> Self {
        SubsurfaceInteraction {
            pi: value.interaction.pi,
            n: value.interaction.n,
            dpdu: value.dpdu,
            dpdv: value.dpdv,
            ns: value.shading.n,
            dpdus: value.shading.dpdu,
            dpdvs: value.shading.dpdv,
        }
    }
}

impl From<&SubsurfaceInteraction> for SurfaceInteraction {
    fn from(value: &SubsurfaceInteraction) -> Self {
        SurfaceInteraction {
            interaction: Interaction::default(),
            dpdu: value.dpdu,
            dpdv: value.dpdv,
            dndu: Normal3f::ZERO,
            dndv: Normal3f::ZERO,
            shading: SurfaceInteractionShading {
                n: value.ns,
                dpdu: value.dpdus,
                dpdv: value.dpdvs,
                dndu: Normal3f::ZERO,
                dndv: Normal3f::ZERO,
            },
            face_index: 0,
            material: None,
            area_light: None,
            dpdx: Vec3f::ZERO,
            dpdy: Vec3f::ZERO,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }
}

pub fn beam_diffusion_ss(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    let sigma_t = sigma_a + sigma_s;
    let rho = sigma_s / sigma_t;
    let t_crit = r * safe::sqrt(sqr(eta) - 1.0);

    let mut ess = 0.0;
    const N_SAMPLES: usize = 100;
    for i in 0..N_SAMPLES {
        let ti = t_crit + sample_exponential((i as Float + 0.5) / N_SAMPLES as Float, sigma_t);
        let d = Float::sqrt(sqr(r) + sqr(ti));
        let cos_theta_o = ti / d;

        ess += rho * Float::exp(-sigma_t * (d + t_crit)) / sqr(d)
            * henyey_greenstein(cos_theta_o, g) * (1.0 - fr_dielectric(-cos_theta_o, eta))
            * cos_theta_o.abs();
    }

    ess / N_SAMPLES as Float
}

pub fn beam_diffusion_ms(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    const N_SAMPLES: usize = 100;

    let mut ed = 0.0;
    let sigmap_s = sigma_s * (1.0 - g);
    let sigmap_t = sigma_a + sigmap_s;
    let rhop = sigmap_s / sigmap_t;

    let d_g = (2.0 * sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t);
    let sigma_tr = safe::sqrt(sigma_a / d_g);

    let fm1 = fresnel_moment_1(eta);
    let fm2 = fresnel_moment_2(eta);
    let ze = -2.0 * d_g * (1.0 + 3.0 * fm2) / (1.0 - 2.0 * fm1);

    let cphi = 0.25 * (1.0 - 2.0 * fm1);
    let ce = 0.5 * (1.0 - 3.0 * fm2);

    for i in 0..N_SAMPLES {
        let zr = sample_exponential((i as Float + 0.5) / N_SAMPLES as Float, sigmap_t);
        let zv = -zr + 2.0 * ze;
        let dr = Float::sqrt(sqr(r) + sqr(zr));
        let dv = Float::sqrt(sqr(r) + sqr(zv));

        // TODO: fast_exp()
        let phi_d = FRAC_1_4PI / d_g * (Float::exp(-sigma_tr * dr) / dr - Float::exp(-sigma_tr * dv) / dv);
        let edn = FRAC_1_4PI * (zr * (1.0 + sigma_tr * dr) * Float::exp(-sigma_tr * dr) / dr.powi(3))
            - zv * (1.0 + sigma_tr * dv) * Float::exp(-sigma_tr * dv) / dv.powi(3);

        let e = phi_d * cphi + edn * ce;
        let kappa = 1.0 - Float::exp(-2.0 * sigmap_t * (dr + zr));
        ed += kappa * rhop * rhop * e;
    }

    ed / N_SAMPLES as Float
}

pub struct BSSRDFTable {
    pub rho_samples: Vec<Float>,
    pub radius_samples: Vec<Float>,
    pub profile: Vec<Float>,
    pub rho_eff: Vec<Float>,
    pub profile_cdf: Vec<Float>,
}

impl BSSRDFTable {
    pub fn eval_profile(&self, rho_index: usize, radius_index: usize) -> Float {
        assert!(rho_index < self.rho_samples.len());
        assert!(radius_index < self.radius_samples.len());
        self.profile[rho_index * self.radius_samples.len() + radius_index]
    }

    pub fn compute_beam_diffusion_bssrdf(&mut self, g: Float, eta: Float) {
        self.radius_samples[0] = 0.0;
        self.radius_samples[1] = 2.5e-3;
        for i in 2..self.radius_samples.len() {
            self.radius_samples[i] = self.radius_samples[i - 1] * 1.2;
        }

        // TODO: fast_exp()
        for i in 0..self.rho_samples.len() {
            self.rho_samples[i] = (1.0 - Float::exp(-8.0 * i as Float / (self.rho_samples.len() - 1) as Float))
                / (1.0 - Float::exp(-8.0));
        }

        // TODO: parallelize this
        for i in 0..self.rho_samples.len() {
            let n_samples = self.radius_samples.len();
            for j in 0..n_samples {
                let rho = self.rho_samples[i];
                let r = self.radius_samples[j];
                self.profile[i * n_samples + j] = 2.0 * PI * r
                    * (beam_diffusion_ss(rho, 1.0 - rho, g, eta, r) 
                    + beam_diffusion_ms(rho, 1.0 - rho, g, eta, r));
            }

            self.rho_eff[i] = integrate_catmull_rom(
                &self.radius_samples,
                &self.profile[i * n_samples..i * n_samples + n_samples],
                &mut self.profile_cdf[i * n_samples..i * n_samples + n_samples],
            );
        }
    }

    /// Returns `(sigma_a, sigma_s)`
    pub fn subsurface_from_diffuse(
        &self,
        rho_eff: &SampledSpectrum,
        mfp: &SampledSpectrum,
    ) -> (SampledSpectrum, SampledSpectrum) {
        let mut sigma_a = SampledSpectrum::from_const(0.0);
        let mut sigma_s = SampledSpectrum::from_const(0.0);
        for c in 0..NUM_SPECTRUM_SAMPLES {
            let rho = invert_catmull_rom(&self.rho_samples, &self.rho_eff, self.rho_eff[c]);
            sigma_s[c] = rho / mfp[c];
            sigma_a[c] = (1.0 - rho) / mfp[c];
        }
        (sigma_a, sigma_s)
    }
}

pub struct BSSRDFProbeSegment {
    pub p0: Point3f,
    pub p1: Point3f,
}

pub struct TabulatedBSSRDF {
    po: Point3f,
    wo: Vec3f,
    ns: Normal3f,
    eta: Float,
    sigma_t: SampledSpectrum,
    rho: SampledSpectrum,
    table: BSSRDFTable, // TODO: Is it better to use a pointer here?
}

impl TabulatedBSSRDF {
    pub fn new(
        po: Point3f,
        ns: Normal3f,
        wo: Vec3f,
        eta: Float,
        sigma_a: &SampledSpectrum,
        sigma_s: &SampledSpectrum,
        table: BSSRDFTable,
    ) -> TabulatedBSSRDF {
        let sigma_t = sigma_a + sigma_s;
        let rho = sigma_s.safe_div(&sigma_t);
        TabulatedBSSRDF {
            po,
            wo,
            ns,
            eta,
            sigma_t,
            rho,
            table,
        }
    }

    pub fn sp(&self, pi: Point3f) -> SampledSpectrum {
        self.sr(self.po.distance(pi))
    }

    pub fn sr(&self, r: Float) -> SampledSpectrum {
        let mut sr = SampledSpectrum::from_const(0.0);

        for i in 0..NUM_SPECTRUM_SAMPLES {
            let r_optical = r * self.sigma_t[i];
            let (mut rho_offset, mut radius_offset) = (0, 0);
            let (mut rho_weights, mut radius_weights) = ([0.0; 4], [0.0; 4]);
            if !catmull_rom_weights(&self.table.rho_samples, self.rho[i], &mut rho_offset, &mut rho_weights)
            || !catmull_rom_weights(&self.table.radius_samples, r_optical, &mut radius_offset, &mut radius_weights) {
                continue;
            }

            let mut sr0 = 0.0;

            for (j, rho_w) in rho_weights.into_iter().enumerate() {
                for (k, radius_w) in radius_weights.into_iter().enumerate() {
                    let weight = rho_w * radius_w;
                    if weight != 0.0 {
                        sr0 += weight * self.table.eval_profile(rho_offset as usize + j, radius_offset as usize + k);
                    }
                }
            }

            if r_optical != 0.0 {
                sr0 /= 2.0 * PI * r_optical;
            }

            sr[i] = sr0;
        }

        sr *= self.sigma_t * self.sigma_t;
        sr.clamp_zero()
    }

    pub fn sample_sr(&self, u: Float) -> Option<Float> {
        if self.sigma_t[0] == 0.0 {
            return None;
        }

        Some(sample_catmull_rom_2d(
            &self.table.rho_samples,
            &self.table.radius_samples,
            &self.table.profile,
            &self.table.profile_cdf,
            self.rho[0],
            u,
            None,
            None,
        ) / self.sigma_t[0])
    }
    
    pub fn pdf_sr(&self, r: Float) -> SampledSpectrum {
        let mut pdf = SampledSpectrum::from_const(0.0);

        for i in 0..NUM_SPECTRUM_SAMPLES {
            let r_optical = r * self.sigma_t[i];

            let (mut rho_offset, mut radius_offset) = (0, 0);
            let (mut rho_weights, mut radius_weights) = ([0.0; 4], [0.0; 4]);
            if !catmull_rom_weights(&self.table.rho_samples, self.rho[i], &mut rho_offset, &mut rho_weights)
            || !catmull_rom_weights(&self.table.radius_samples, r_optical, &mut radius_offset, &mut radius_weights) {
                continue;
            }

            let mut sr = 0.0;
            let mut rho_eff = 0.0;
            
            for (j, rho_w) in rho_weights.into_iter().enumerate() {
                if rho_w != 0.0 {
                    rho_eff += self.table.rho_eff[rho_offset as usize + j] * rho_w;
                    for (k, radius_w) in radius_weights.into_iter().enumerate() {
                        if radius_w != 0.0 {
                            sr += self.table.eval_profile(rho_offset as usize + j, radius_offset as usize + k)
                                * rho_w * radius_w;
                        }
                    }
                }
            }

            if r_optical != 0.0 {
                sr /= 2.0 * PI * r_optical;
            }

            pdf[i] = sr * sqr(self.sigma_t[i]) / rho_eff;
        }

        pdf.clamp_zero()
    }

    pub fn pdf_sp(&self, pi: Point3f, ni: Normal3f) -> SampledSpectrum {
        let d = pi - self.po;
        let f = Frame::from_z(self.ns.into());
        let d_local = f.localize(d.into());
        let n_local = f.localize(ni.into());

        let r_proj = [
            Float::sqrt(sqr(d_local.y) + sqr(d_local.z)),
            Float::sqrt(sqr(d_local.z) + sqr(d_local.x)),
            Float::sqrt(sqr(d_local.x) + sqr(d_local.y)),
        ];

        let mut pdf = SampledSpectrum::from_const(0.0);
        let axis_prob = [0.25, 0.25, 0.5];
        for (axis, prob) in axis_prob.into_iter().enumerate() {
            pdf += self.pdf_sr(r_proj[axis]) * n_local[axis].abs() * prob;
        }

        pdf
    }
}

impl AbstractBSSRDF for TabulatedBSSRDF {
    type BxDF = NormalizedFresnelBxDF;

    fn sample_sp(&self, u1: Float, u2: Point2f) -> Option<BSSRDFProbeSegment> {
        let f = if u1 < 0.25 {
            Frame::from_x(self.ns.into())
        } else if u1 < 0.5 {
            Frame::from_y(self.ns.into())
        } else {
            Frame::from_z(self.ns.into())
        };

        let r = self.sample_sr(u2.x)?;
        let phi = 2.0 * PI * u2.y;
        let r_max = self.sample_sr(0.999)?;

        if r >= r_max {
            return None;
        }

        let l = 2.0 * Float::sqrt(r_max * r_max - r * r);
        let p_start = self.po + r * (f.x * phi.cos() + f.y * phi.sin()) - l * f.z / 2.0;
        let p_target = p_start + l * f.z;

        Some(BSSRDFProbeSegment { p0: p_start, p1: p_target })
    }

    fn probe_intersection_to_sample(&self, si: &SubsurfaceInteraction) -> BSSRDFSample {
        let bxdf = NormalizedFresnelBxDF::new(self.eta);
        let wo = Vec3f::from(si.ns);
        let bsdf = BSDF::new(si.ns, si.dpdus, BxDF::NormalizedFresnel(bxdf));
        BSSRDFSample { sp: self.sp(si.p()), pdf: self.pdf_sp(si.p(), si.n), sw: bsdf, wo }
    }
}
