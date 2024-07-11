// Pbrt 3.11 Interactions

use crate::math::*;

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
    // pub material: Option<Arc<Material>>,
    // pub area_light: Option<Arc<Light>>,
    pub material: Option<()>,
    pub area_light: Option<()>,
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
            self.shading.dpdu = self.shading.dpdu / 1e8;
            self.shading.dpdv = self.shading.dpdv / 1e8;
        }
    }
}


/* 
pub struct MediumInteraction {
    pub interaction: Interaction,
    pub phase: PhaseFunction,
}
*/
