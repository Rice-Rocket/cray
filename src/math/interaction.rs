// Pbrt 3.11 Interactions

use std::sync::Arc;

use rand::rngs::SmallRng;
use vect::Dot;

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{diffuse::DiffuseBxDF, BxDF, BxDFFlags}, camera::{AbstractCamera, Camera}, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, light::{AbstractLight, Light}, material::{self, AbstractMaterial, Material, MaterialEvalContext, UniversalTextureEvaluator}, math::*, media::{Medium, MediumInterface}, numeric::DifferenceOfProducts, options::Options, phase::PhaseFunction, sampler::{AbstractSampler as _, Sampler}};

#[derive(Debug, Clone, Default)]
pub struct Interaction {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Vec3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub medium: Option<Arc<Medium>>,
    pub medium_interface: Option<Arc<MediumInterface>>,
}

impl Interaction {
    pub fn new(pi: Point3fi, n: Normal3f, uv: Point2f, wo: Vec3f, time: Float) -> Self {
        Self { pi, time, wo, n, uv, medium: None, medium_interface: None }
    }

    pub fn position(&self) -> Point3f {
        self.pi.into()
    }

    pub fn offset_ray_origin_d(&self, w: Vec3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    pub fn offset_ray_origin_p(&self, p: Point3f) -> Point3f {
        self.offset_ray_origin_d((p - self.position()).into())
    }
    
    pub fn spawn_ray(&self, d: Vec3f) -> RayDifferential {
        RayDifferential::new(Ray::new_with_medium(self.offset_ray_origin_d(d), d, self.get_medium_from_w(d)), None)
    }

    pub fn spawn_ray_to(&self, p: Point3f) -> Ray {
        let mut r = Ray::spawn_ray_to(self.pi, self.n, self.time, p);
        r.medium = self.get_medium_from_w(r.direction);
        r
    }

    pub fn spawn_ray_to_interaction(&self, it: &Self) -> Ray {
        let mut r = Ray::spawn_ray_to_both_offset(self.pi, self.n, self.time, it.pi, it.n);
        r.medium = self.get_medium_from_w(r.direction);
        r
    }

    pub fn get_medium_from_w(&self, w: Vec3f) -> Option<Arc<Medium>> {
        if let Some(ref mi) = self.medium_interface {
            if w.dot(self.n) > 0.0 { mi.outside.clone() } else { mi.inside.clone() }
        } else {
            self.medium.clone()
        }
    }

    pub fn get_medium(&self) -> Option<Arc<Medium>> {
        if let Some(ref mi) = self.medium_interface {
            debug_assert_eq!(mi.inside, mi.outside);
            mi.inside.clone()
        } else {
            self.medium.clone()
        }
    }

    pub fn is_surface_interaction(&self) -> bool {
        self.n != Normal3f::ZERO
    }

    pub fn is_medium_interaction(&self) -> bool {
        !self.is_surface_interaction()
    }
}

#[derive(Debug, Clone)]
pub enum GeneralInteraction {
    Surface(SurfaceInteraction),
    Medium(MediumInteraction),
}

impl GeneralInteraction {
    #[inline]
    pub fn intr(&self) -> &Interaction {
        match self {
            GeneralInteraction::Surface(i) => &i.interaction,
            GeneralInteraction::Medium(i) => &i.interaction,
        }
    }
    
    #[inline]
    pub fn intr_mut(&mut self) -> &mut Interaction {
        match self {
            GeneralInteraction::Surface(i) => &mut i.interaction,
            GeneralInteraction::Medium(i) => &mut i.interaction,
        }
    }

    #[inline]
    pub fn as_surface(&self) -> &SurfaceInteraction {
        match self {
            GeneralInteraction::Surface(i) => i,
            GeneralInteraction::Medium(_) => panic!("assumed surface interaction but was medium interaction."),
        }
    }

    #[inline]
    pub fn as_medium(&self) -> &MediumInteraction {
        match self {
            GeneralInteraction::Surface(_) => panic!("assumed medium interaction but was surface interaction."),
            GeneralInteraction::Medium(i) => i,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceInteractionShading {
    pub n: Normal3f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Debug, Clone)]
pub struct SurfaceInteraction {
    pub interaction: Interaction,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: SurfaceInteractionShading,
    pub face_index: i32,
    pub material: Option<Arc<Material>>,
    pub area_light: Option<Arc<Light>>,
    pub dpdx: Vec3f,
    pub dpdy: Vec3f,
    pub dudx: Float,
    pub dvdx: Float,
    pub dudy: Float,
    pub dvdy: Float,
}

impl SurfaceInteraction {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pi: Point3fi,
        uv: Point2f,
        wo: Vec3f,
        dpdu: Vec3f,
        dpdv: Vec3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        flip_normal: bool,
    ) -> SurfaceInteraction {
        let norm_sign = if flip_normal { -1.0 } else { 1.0 };
        let n = Normal3f::from(dpdu.cross(dpdv).normalize()) * norm_sign;
        let interaction = Interaction::new(pi, n, uv, wo, time);

        SurfaceInteraction {
            interaction,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shading: SurfaceInteractionShading {
                n,
                dpdu,
                dpdv,
                dndu,
                dndv,
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

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_face_index(
        pi: Point3fi,
        uv: Point2f,
        wo: Vec3f,
        dpdu: Vec3f,
        dpdv: Vec3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        flip_normal: bool,
        face_index: i32,
    ) -> SurfaceInteraction {
        SurfaceInteraction {
            face_index,
            ..SurfaceInteraction::new(
                pi,
                uv,
                wo,
                dpdu,
                dpdv,
                dndu,
                dndv,
                time,
                flip_normal
            )
        }
    }

    pub fn position(&self) -> Point3f {
        self.interaction.position()
    }

    pub fn set_intersection_properties(
        &mut self,
        material: &Arc<Material>,
        area_light: Option<&Arc<Light>>,
        prim_medium_interface: Option<&Arc<MediumInterface>>,
        ray_medium: Option<&Arc<Medium>>,
    ) {
        self.area_light = area_light.cloned();
        self.material = Some(material.clone());

        if prim_medium_interface.as_ref().is_some_and(|m| m.is_transition()) {
            self.interaction.medium_interface = prim_medium_interface.cloned();
        } else {
            self.interaction.medium = ray_medium.cloned();
        }
    }

    pub fn get_bsdf(
        &mut self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        camera: &Camera,
        sampler: &mut Sampler,
        options: &Options,
        rng: &mut SmallRng,
    ) -> Option<BSDF> {
        self.compute_differentials(ray, camera, sampler.samples_per_pixel(), options);

        let mut material = self.material.as_ref().map(|m| m.clone())?;

        let mut is_mixed = matches!(material.as_ref(), Material::Mix(_));

        let mut material_eval_context = MaterialEvalContext::from(&*self);

        while is_mixed {
            match material.as_ref() {
                Material::Mix(m) => {
                    material = m.choose_material(&UniversalTextureEvaluator, &material_eval_context, rng);
                },
                _ => is_mixed = false,
            };
        }

        let material = match material.as_ref() {
            Material::Interface => return None,
            Material::Single(m) => m,
            Material::Mix(m) => unreachable!(),
        };

        let displacement = material.get_displacement();
        let normal_map = material.get_normal_map();
        if displacement.is_some() || normal_map.is_some() {
            let (dpdu, dpdv) = if let Some(displacement) = displacement {
                material::bump_map(
                    UniversalTextureEvaluator,
                    displacement,
                    &self.into(),
                )
            } else {
                let normal_map = normal_map.unwrap();
                material::normal_map(
                    normal_map.as_ref(),
                    &self.into(),
                )
            };

            let ns = dpdu.cross(dpdv).normalize();
            self.set_shading_geometry(ns.into(), dpdu, dpdv, self.shading.dndu, self.shading.dndv, false);
            material_eval_context = MaterialEvalContext::from(&*self);
        }

        let bsdf = material.get_bsdf(
            &UniversalTextureEvaluator,
            &material_eval_context,
            lambda,
        );

        let bsdf = if options.force_diffuse {
            let r = bsdf.rho_hd(
                self.interaction.wo,
                &[sampler.get_1d()],
                &[sampler.get_2d()],
            );

            BSDF::new(
                self.shading.n,
                self.shading.dpdu,
                BxDF::Diffuse(DiffuseBxDF::new(r)),
            )
        } else {
            bsdf
        };

        Some(bsdf)
    }

    pub fn get_bssrdf(
        &self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        camera: &Camera,
        rng: &mut SmallRng,
    ) -> Option<BSSRDF> {
        let mut material = self.material.as_ref().map(|m| m.clone())?;
        let mut is_mixed = matches!(material.as_ref(), Material::Mix(_));
        let mut material_eval_context = MaterialEvalContext::from(self);

        while is_mixed {
            match material.as_ref() {
                Material::Mix(m) => {
                    material = m.choose_material(&UniversalTextureEvaluator, &material_eval_context, rng);
                },
                _ => is_mixed = false,
            };
        }

        let material = match material.as_ref() {
            Material::Interface => return None,
            Material::Single(m) => m,
            Material::Mix(m) => unreachable!(),
        };

        material.get_bssrdf(&UniversalTextureEvaluator, &material_eval_context, lambda)
    }

    pub fn compute_differentials(
        &mut self,
        ray: &RayDifferential,
        camera: &Camera,
        samples_per_pixel: i32,
        options: &Options,
    ) {
        if options.disable_texture_filtering {
            self.dudx = 0.0;
            self.dudy = 0.0;
            self.dvdx = 0.0;
            self.dvdy = 0.0;
            self.dpdx = Vec3f::ZERO;
            self.dpdy = Vec3f::ZERO;
            return;
        }

        if ray.aux.as_ref().is_some_and(|aux| {
            self.interaction.n.dot(aux.rx_direction) != 0.0
                && self.interaction.n.dot(aux.ry_direction) != 0.0
        }) {
            let aux = ray.aux.as_ref().unwrap();
            let d = -self.interaction.n.dot(self.position());
            let tx = (-self.interaction.n.dot(aux.rx_origin) - d)
                / self.interaction.n.dot(aux.rx_direction);
            debug_assert!(tx.is_finite() && !tx.is_nan());
            let px = aux.rx_origin + tx * aux.rx_direction;

            let ty = (-self.interaction.n.dot(aux.ry_origin) - d)
                / self.interaction.n.dot(aux.ry_direction);
            debug_assert!(ty.is_finite() && !ty.is_nan());
            let py = aux.ry_origin + ty * aux.ry_direction;

            self.dpdx = (px - self.position()).into();
            self.dpdy = (py - self.position()).into();
        } else {
            (self.dpdx, self.dpdy) = camera.approximate_dp_dxy(
                self.position(),
                self.interaction.n,
                self.interaction.time,
                samples_per_pixel,
                options,
            );
        }
        
        let ata00 = self.dpdu.dot(self.dpdu);
        let ata01 = self.dpdu.dot(self.dpdv);
        let ata11 = self.dpdv.dot(self.dpdv);
        let inv_det = 1.0 / Float::difference_of_products(ata00, ata11, ata01, ata01);
        let inv_det = if inv_det.is_finite() { inv_det } else { 0.0 };

        let atb0x = self.dpdu.dot(self.dpdx);
        let atb1x = self.dpdv.dot(self.dpdx);
        let atb0y = self.dpdu.dot(self.dpdy);
        let atb1y = self.dpdv.dot(self.dpdy);

        self.dudx = Float::difference_of_products(ata11, atb0x, ata01, atb1x) * inv_det;
        self.dvdx = Float::difference_of_products(ata00, atb1x, ata01, atb0x) * inv_det;
        self.dudy = Float::difference_of_products(ata11, atb0y, ata01, atb1y) * inv_det;
        self.dvdy = Float::difference_of_products(ata00, atb1y, ata01, atb0y) * inv_det;

        self.dudx = if self.dudx.is_finite() {
            Float::clamp(self.dudx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdx = if self.dvdx.is_finite() {
            Float::clamp(self.dvdx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dudy = if self.dudy.is_finite() {
            Float::clamp(self.dudy, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdy = if self.dvdy.is_finite() {
            Float::clamp(self.dvdy, -1e8, 1e8)
        } else {
            0.0
        };
    }

    /// Computes the emitted radiance
    pub fn le(&self, w: Vec3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        if let Some(ref area_light) = self.area_light {
            area_light.as_ref().l(self.position(), self.interaction.n, self.interaction.uv, w, lambda)
        } else {
            SampledSpectrum::from_const(0.0)
        }
    }

    pub fn set_shading_geometry(
        &mut self,
        ns: Normal3f,
        dpdus: Vec3f,
        dpdvs: Vec3f,
        dndus: Normal3f,
        dndvs: Normal3f,
        orientation_is_authoritative: bool,
    ) {
        self.shading.n = ns;
        debug_assert!(self.shading.n != Normal3f::ZERO);
        if orientation_is_authoritative {
            self.interaction.n = self.interaction.n.facing(self.shading.n);
        } else {
            self.shading.n = self.shading.n.facing(self.interaction.n);
        }

        self.shading.dpdu = dpdus;
        self.shading.dpdv = dpdvs;
        self.shading.dndu = dndus;
        self.shading.dndv = dndvs;

        while self.shading.dpdu.length_squared() > 1e16 || self.shading.dpdv.length_squared() > 1e16 {
            self.shading.dpdu /= 1e8;
            self.shading.dpdv /= 1e8;
        }
    }

    pub fn skip_intersection(&mut self, ray: &mut RayDifferential, t: Float) {
        let mut new_ray = self.interaction.spawn_ray(ray.ray.direction);
        new_ray.aux = if let Some(aux) = &ray.aux {
            let rx_origin = aux.rx_origin + t * aux.rx_direction;
            let ry_origin = aux.ry_origin + t * aux.ry_direction;
            Some(AuxiliaryRays {
                rx_origin,
                ry_origin,
                rx_direction: aux.rx_direction,
                ry_direction: aux.ry_direction,
            })
        } else {
            None
        };

        *ray = new_ray
    }

    pub fn spawn_ray_with_differentials(
        &self,
        ray_i: &RayDifferential,
        wi: Vec3f,
        flags: BxDFFlags,
        eta: Float
    ) -> RayDifferential {
        let mut rd = self.interaction.spawn_ray(wi);
        if let Some(aux) = &ray_i.aux {
            let mut n = self.shading.n;
            let mut dndx = self.shading.dndu * self.dudx + self.shading.dndv * self.dvdx;
            let mut dndy = self.shading.dndu * self.dudy + self.shading.dndv * self.dvdy;
            let dwodx = -aux.rx_direction - self.interaction.wo;
            let dwody = -aux.ry_direction - self.interaction.wo;

            rd.aux = if flags == BxDFFlags::SPECULAR_REFLECTION {
                let rx_origin = self.interaction.position() + self.dpdx;
                let ry_origin = self.interaction.position() + self.dpdy;

                let dwo_dotn_dx = dwodx.dot(n) + self.interaction.wo.dot(dndx);
                let dwo_dotn_dy = dwody.dot(n) + self.interaction.wo.dot(dndy);

                let rx_direction = wi - dwodx + 2.0 * (self.interaction.wo.dot(n) * dndx + dwo_dotn_dx * n);
                let ry_direction = wi - dwody + 2.0 * (self.interaction.wo.dot(n) * dndy + dwo_dotn_dy * n);

                Some(AuxiliaryRays {
                    rx_origin,
                    rx_direction,
                    ry_origin,
                    ry_direction,
                })
            } else if flags == BxDFFlags::SPECULAR_TRANSMISSION {
                let rx_origin = self.interaction.position() + self.dpdx;
                let ry_origin = self.interaction.position() + self.dpdy;

                if self.interaction.wo.dot(n) < 0.0 {
                    n = -n;
                    dndx = -dndx;
                    dndy = -dndy;
                }

                let dwo_dotn_dx = dwodx.dot(n) + self.interaction.wo.dot(dndx);
                let dwo_dotn_dy = dwody.dot(n) + self.interaction.wo.dot(dndy);

                let mu = self.interaction.wo.dot(n) / eta - wi.dot(n).abs();
                let dmudx = dwo_dotn_dx * (1.0 / eta + 1.0 / sqr(eta) * self.interaction.wo.dot(n) / wi.dot(n));
                let dmudy = dwo_dotn_dy * (1.0 / eta + 1.0 / sqr(eta) * self.interaction.wo.dot(n) / wi.dot(n));

                let rx_direction = wi - eta * dwodx + Vec3f::from(mu * dndx + dmudx * n);
                let ry_direction = wi - eta * dwody + Vec3f::from(mu * dndy + dmudy * n);

                Some(AuxiliaryRays {
                    rx_origin,
                    rx_direction,
                    ry_origin,
                    ry_direction,
                })
            } else {
                None
            };
        }

        if rd.aux.as_ref().is_some_and(|aux| {
            aux.rx_direction.length_squared() > 1e16 || aux.ry_direction.length_squared() > 1e16
            || aux.rx_origin.length_squared() > 1e16 || aux.ry_origin.length_squared() > 1e16
        }) {
            rd.aux = None;
        }

        rd
    }
}

#[derive(Debug, Clone)]
pub struct MediumInteraction {
    pub interaction: Interaction,
    pub phase: PhaseFunction,
}

impl MediumInteraction {
    pub fn new(
        pi: Point3fi,
        wo: Vec3f,
        time: Float,
        medium: Option<Arc<Medium>>,
        phase: PhaseFunction,
    ) -> MediumInteraction {
        MediumInteraction {
            interaction: Interaction {
                pi,
                time,
                wo,
                n: Normal3f::ZERO,
                uv: Point2f::ZERO,
                medium,
                medium_interface: None,
            },
            phase,
        }
    }
}
