use std::{collections::HashMap, sync::Arc};

use conductor::ConductorMaterial;
use dielectric::DielectricMaterial;
use diffuse::DiffuseMaterial;
use rand::{rngs::SmallRng, Rng};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{AbstractBxDF, BxDF}, color::{sampled::SampledSpectrum, spectrum::Spectrum, wavelengths::SampledWavelengths}, image::{Image, WrapMode, WrapMode2D}, interaction::SurfaceInteraction, reader::{paramdict::{NamedTextures, TextureParameterDictionary}, target::FileLoc}, texture::{AbstractFloatTexture, AbstractSpectrumTexture, FloatTexture, SpectrumTexture, TextureEvalContext}, Float, Frame, Normal3f, Point2f, Point3f, Vec2f, Vec3f};

pub mod diffuse;
pub mod conductor;
pub mod dielectric;

pub trait AbstractMaterial {
    type ConcreteBxDF: AbstractBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF;

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF;

    fn get_bssrdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Option<BSSRDF>;

    fn can_evaluate_textures<T: AbstractTextureEvaluator>(&self, tex_eval: &T) -> bool;

    fn get_normal_map(&self) -> Option<Arc<Image>>;

    fn get_displacement(&self) -> Option<Arc<FloatTexture>>;

    fn has_subsurface_scattering(&self) -> bool;
}

#[derive(Debug, Clone)]
pub enum Material {
    Interface,
    Single(SingleMaterial),
    Mix(MixMaterial),
}

impl Material {
    pub fn create(
        name: &str,
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        named_materials: &HashMap<String, Arc<Material>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> Material {
        match name {
            "interface" => {
                Material::Interface
            },
            "mix" => {
                let material_names = parameters.get_string_array("materials");
                if material_names.len() != 2 {
                    panic!("expected exactly two materials for mix material");
                }

                let named_material: Vec<Arc<Material>> = material_names.iter()
                    .map(|name| {
                        let material = named_materials.get(name).expect("material not found");
                        material.clone()
                    })
                    .collect();

                let materials: [Arc<Material>; 2] = [
                    named_material[0].clone(),
                    named_material[1].clone()
                ];

                Material::Mix(MixMaterial::create(materials, parameters, loc, textures))
            },
            _ => Material::Single(SingleMaterial::create(
                name,
                parameters,
                textures,
                normal_map,
                named_materials,
                cached_spectra,
                loc,
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub enum SingleMaterial {
    Diffuse(DiffuseMaterial),
    Conductor(ConductorMaterial),
    Dielectric(DielectricMaterial),
}

impl SingleMaterial {
    pub fn create(
        name: &str,
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        named_materials: &HashMap<String, Arc<Material>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> SingleMaterial {
        match name {
            "diffuse" => SingleMaterial::Diffuse(DiffuseMaterial::create(
                parameters,
                textures,
                normal_map,
                cached_spectra,
                loc,
            )),
            "conductor" => SingleMaterial::Conductor(ConductorMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            "dielectric" => SingleMaterial::Dielectric(DielectricMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            _ => panic!("material {} unknown", name),
        }
    }
}

impl AbstractMaterial for SingleMaterial {
    type ConcreteBxDF = BxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        match self {
            SingleMaterial::Diffuse(m) => BxDF::Diffuse(m.get_bxdf(tex_eval, ctx, lambda)),
            SingleMaterial::Conductor(m) => BxDF::Conductor(m.get_bxdf(tex_eval, ctx, lambda)),
            SingleMaterial::Dielectric(m) => BxDF::Dielectric(m.get_bxdf(tex_eval, ctx, lambda)),
        }
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        match self {
            SingleMaterial::Diffuse(m) => m.get_bsdf(tex_eval, ctx, lambda),
            SingleMaterial::Conductor(m) => m.get_bsdf(tex_eval, ctx, lambda),
            SingleMaterial::Dielectric(m) => m.get_bsdf(tex_eval, ctx, lambda),
        }
    }

    fn get_bssrdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Option<BSSRDF> {
        match self {
            SingleMaterial::Diffuse(m) => m.get_bssrdf(tex_eval, ctx, lambda),
            SingleMaterial::Conductor(m) => m.get_bssrdf(tex_eval, ctx, lambda),
            SingleMaterial::Dielectric(m) => m.get_bssrdf(tex_eval, ctx, lambda),
        }
    }

    fn can_evaluate_textures<T: AbstractTextureEvaluator>(&self, tex_eval: &T) -> bool {
        match self {
            SingleMaterial::Diffuse(m) => m.can_evaluate_textures(tex_eval),
            SingleMaterial::Conductor(m) => m.can_evaluate_textures(tex_eval),
            SingleMaterial::Dielectric(m) => m.can_evaluate_textures(tex_eval),
        }
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        match self {
            SingleMaterial::Diffuse(m) => m.get_normal_map(),
            SingleMaterial::Conductor(m) => m.get_normal_map(),
            SingleMaterial::Dielectric(m) => m.get_normal_map(),
        }
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        match self {
            SingleMaterial::Diffuse(m) => m.get_displacement(),
            SingleMaterial::Conductor(m) => m.get_displacement(),
            SingleMaterial::Dielectric(m) => m.get_displacement(),
        }
    }

    fn has_subsurface_scattering(&self) -> bool {
        match self {
            SingleMaterial::Diffuse(m) => m.has_subsurface_scattering(),
            SingleMaterial::Conductor(m) => m.has_subsurface_scattering(),
            SingleMaterial::Dielectric(m) => m.has_subsurface_scattering(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MixMaterial {
    amount: Arc<FloatTexture>,
    materials: [Arc<Material>; 2],
}

impl MixMaterial {
    pub fn create(
        materials: [Arc<Material>; 2],
        parameters: &mut TextureParameterDictionary,
        _loc: &FileLoc,
        textures: &NamedTextures,
    ) -> MixMaterial {
        let amount = parameters.get_float_texture("amount", 0.5, textures);
        MixMaterial { amount, materials }
    }

    pub fn choose_material(&self, tex_eval: &UniversalTextureEvaluator, ctx: &MaterialEvalContext, rng: &mut SmallRng) -> Arc<Material> {
        let amt = tex_eval.evaluate_float(&self.amount, &ctx.tex_ctx);

        if amt <= 0.0 {
            return self.materials[0].clone()
        } else if amt >= 1.0 {
            return self.materials[1].clone()
        }

        let u = rng.gen::<Float>();
        if amt < u {
            self.materials[0].clone()
        } else {
            self.materials[1].clone()
        }
    }
}

pub struct MaterialEvalContext {
    tex_ctx: TextureEvalContext,
    wo: Vec3f,
    ns: Normal3f,
    dpdus: Vec3f,
}

impl From<&SurfaceInteraction> for MaterialEvalContext {
    fn from(si: &SurfaceInteraction) -> Self {
        MaterialEvalContext {
            tex_ctx: TextureEvalContext::from(si),
            wo: si.interaction.wo,
            ns: si.shading.n,
            dpdus: si.shading.dpdu,
        }
    }
}

pub trait AbstractTextureEvaluator {
    fn can_evaluate(&self, f_tex: &[Option<Arc<FloatTexture>>], s_tex: &[Option<Arc<SpectrumTexture>>]) -> bool;

    fn evaluate_float(&self, tex: &Arc<FloatTexture>, ctx: &TextureEvalContext) -> Float;

    fn evaluate_spectrum(
        &self,
        tex: &Arc<SpectrumTexture>,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
}

pub struct UniversalTextureEvaluator;

impl AbstractTextureEvaluator for UniversalTextureEvaluator {
    fn can_evaluate(&self, _f_tex: &[Option<Arc<FloatTexture>>], _s_tex: &[Option<Arc<SpectrumTexture>>]) -> bool {
        true
    }

    fn evaluate_float(&self, tex: &Arc<FloatTexture>, ctx: &TextureEvalContext) -> Float {
        tex.evaluate(ctx)
    }

    fn evaluate_spectrum(
            &self,
            tex: &Arc<SpectrumTexture>,
            ctx: &TextureEvalContext,
            lambda: &SampledWavelengths,
        ) -> SampledSpectrum {
        tex.evaluate(ctx, lambda)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct NormalBumpEvalContextShading {
    pub n: Normal3f,
    pub dpdu: Vec3f,
    pub dpdv: Vec3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Debug, Copy, Clone)]
pub struct NormalBumpEvalContext {
    pub p: Point3f,
    pub uv: Point2f,
    pub n: Normal3f,
    pub shading: NormalBumpEvalContextShading,
    pub dudx: Float,
    pub dudy: Float,
    pub dvdx: Float,
    pub dvdy: Float,
    pub dpdx: Vec3f,
    pub dpdy: Vec3f,
    pub face_index: i32,
}

impl From<&mut SurfaceInteraction> for NormalBumpEvalContext {
    fn from(value: &mut SurfaceInteraction) -> Self {
        NormalBumpEvalContext {
            p: value.position(),
            uv: value.interaction.uv,
            n: value.shading.n,
            shading: NormalBumpEvalContextShading {
                n: value.shading.n,
                dpdu: value.shading.dpdu,
                dpdv: value.shading.dpdv,
                dndu: value.shading.dndu,
                dndv: value.shading.dndv,
            },
            dudx: value.dudx,
            dudy: value.dudy,
            dvdx: value.dvdx,
            dvdy: value.dvdy,
            dpdx: value.dpdx,
            dpdy: value.dpdy,
            face_index: value.face_index,
        }
    }
}


pub fn normal_map(normal_map: &Image, ctx: &NormalBumpEvalContext) -> (Vec3f, Vec3f) {
    let wrap: WrapMode2D = WrapMode::Repeat.into();
    let uv = Point2f::new(ctx.uv.x, 1.0 - ctx.uv.y);
    let ns = Vec3f::new(
        2.0 * normal_map.bilinear_channel_wrapped(uv, 0, wrap) - 1.0,
        2.0 * normal_map.bilinear_channel_wrapped(uv, 1, wrap) - 1.0,
        2.0 * normal_map.bilinear_channel_wrapped(uv, 2, wrap) - 1.0,
    );

    let ns = ns.normalize();

    let frame = Frame::from_xz(ctx.shading.dpdu.normalize(), ctx.shading.n.into());
    let ns = frame.from_local(ns);

    let ulen = ctx.shading.dpdu.length();
    let vlen = ctx.shading.dpdv.length();
    let dpdu = ctx.shading.dpdu.gram_schmidt(ns).normalize() * ulen;
    let dpdv = ns.cross(dpdu).normalize() * vlen;

    (dpdu, dpdv)
}

pub fn bump_map(tex_eval: UniversalTextureEvaluator, displacement: Arc<FloatTexture>, ctx: &NormalBumpEvalContext) -> (Vec3f, Vec3f) {
    let mut shifted_ctx = *ctx;

    let mut du = 0.5 * (ctx.dudx.abs() + ctx.dudy.abs());
    if du == 0.0 {
        du = 0.0005;
    }

    shifted_ctx.p = ctx.p + du * ctx.shading.dpdu;
    shifted_ctx.uv = ctx.uv + Vec2f::new(du, 0.0);

    let u_displace = tex_eval.evaluate_float(&displacement, &(&shifted_ctx).into());

    let mut dv = 0.5 * (ctx.dvdx.abs() + ctx.dvdy.abs());
    if dv == 0.0 {
        dv = 0.0005;
    }

    shifted_ctx.p = ctx.p + dv * ctx.shading.dpdv;
    shifted_ctx.uv = ctx.uv + Vec2f::new(0.0, dv);
    
    let v_displace = tex_eval.evaluate_float(&displacement, &(&shifted_ctx).into());
    let displace = tex_eval.evaluate_float(&displacement, &ctx.into());

    let dpdu = ctx.shading.dpdu + (u_displace - displace) / du * ctx.shading.n + displace * ctx.shading.dndu;
    let dpdv = ctx.shading.dpdv + (v_displace - displace) / dv * ctx.shading.n + displace * ctx.shading.dndv;

    (dpdu, dpdv)
}
