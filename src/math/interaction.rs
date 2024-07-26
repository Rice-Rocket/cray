// Pbrt 3.11 Interactions

use std::sync::Arc;

use crate::{camera::{Camera, CameraLike}, color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, light::Light, material::Material, math::*, numeric::DifferenceOfProducts, options::Options};

#[derive(Debug, Clone)]
pub struct Interaction {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Vec3f,
    pub n: Normal3f,
    pub uv: Point2f,
}

impl Interaction {
    pub fn new(pi: Point3fi, n: Normal3f, uv: Point2f, wo: Vec3f, time: Float) -> Self {
        Self { pi, time, wo, n, uv }
    }

    pub fn position(&self) -> Point3f {
        self.pi.into()
    }

    fn offset_ray_origin_d(&self, w: Vec3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    fn offset_ray_origin_p(&self, p: Point3f) -> Point3f {
        self.offset_ray_origin_d((p - self.position()).into())
    }
    
    fn spawn_ray(&self, d: Vec3f) -> RayDifferential {
        RayDifferential::new(Ray::new(self.offset_ray_origin_d(d), d), None)
    }

    fn spawn_ray_to(&self, p: Point3f) -> Ray {
        Ray::spawn_ray_to(self.pi, self.n, self.time, p)
    }

    fn spawn_ray_to_interaction(&self, it: &Self) -> Ray {
        Ray::spawn_ray_to_both_offset(self.pi, self.n, self.time, it.pi, it.n)
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

    pub fn set_intersection_properties(&mut self, material: &Arc<Material>, area_light: &Option<Arc<Light>>) {
        self.area_light.clone_from(area_light);
        self.material = Some(material.clone());
    }

    pub fn compute_differentials(
        &mut self,
        ray: RayDifferential,
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
            self.interaction.n.dot(aux.rx_direction.into()) != 0.0
                && self.interaction.n.dot(aux.ry_direction.into()) != 0.0
        }) {
            let aux = ray.aux.as_ref().unwrap();
            let d = -self.interaction.n.dot(self.position().into());
            let tx = (-self.interaction.n.dot(aux.rx_origin.into()) - d)
                / self.interaction.n.dot(aux.rx_direction.into());
            debug_assert!(tx.is_finite() && !tx.is_nan());
            let px = aux.rx_origin + tx * aux.rx_direction;

            let ty = (-self.interaction.n.dot(aux.ry_origin.into()) - d)
                / self.interaction.n.dot(aux.ry_direction.into());
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
            // TODO: Change this when light is implemented
            SampledSpectrum::from_const(0.0)
            // area_light.as_ref().l(self.position(), self.interaction.n, self.interaction.uv, w, lambda)
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
}


/* 
pub struct MediumInteraction {
    pub interaction: Interaction,
    pub phase: PhaseFunction,
}
*/
