use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{diffuse::DiffuseBxDF, BxDF}, color::{rgb_xyz::Rgb, sampled::SampledSpectrum, spectrum::{ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, error, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, texture::{AbstractTextureMapping2D, FloatTexture, SpectrumConstantTexture, SpectrumTexture}, Vec2f};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub enum DebugMaterialMode {
    Normal,
    Position,
    Wo,
    NormalShading,
    Dpdus,
    Dpdx,
    Dpdy,
    Dudxy,
    Dvdxy,
    Uv,
    St,
    Texture,
}

#[derive(Debug, Clone)]
pub struct DebugMaterial {
    color: Arc<SpectrumTexture>,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    mode: DebugMaterialMode,
}

impl DebugMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> DebugMaterial {
        let color = parameters.get_spectrum_texture(
            "color",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        );

        let color = if let Some(color) = color {
            color
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(
                Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5))),
            )))
        };

        let displacement = Some(parameters.get_float_texture("displacement", 0.0, textures));

        let mode = match parameters.get_one_string("mode", "normal").as_str() {
            "normal" => DebugMaterialMode::Normal,
            "position" => DebugMaterialMode::Position,
            "uv" => DebugMaterialMode::Uv,
            "wo" => DebugMaterialMode::Wo,
            "ns" => DebugMaterialMode::NormalShading,
            "dpdus" => DebugMaterialMode::Dpdus,
            "dpdx" => DebugMaterialMode::Dpdx,
            "dpdy" => DebugMaterialMode::Dpdy,
            "dudxy" => DebugMaterialMode::Dudxy,
            "dvdxy" => DebugMaterialMode::Dvdxy,
            "st" => DebugMaterialMode::St,
            "texture" => DebugMaterialMode::Texture,
            s => { error!(loc, "unknown debug material mode '{}'", s); },
        };

        DebugMaterial::new(color, displacement, normal_map, mode)
    }

    pub fn new(
        color: Arc<SpectrumTexture>,
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        mode: DebugMaterialMode,
    ) -> DebugMaterial {
        DebugMaterial {
            color,
            displacement,
            normal_map,
            mode,
        }
    }

    pub fn get_color(&self, ctx: &MaterialEvalContext) -> Rgb {
        match self.mode {
            DebugMaterialMode::Normal => Rgb::new(ctx.tex_ctx.n.x, ctx.tex_ctx.n.y, ctx.tex_ctx.n.z),
            DebugMaterialMode::Position => Rgb::new(ctx.tex_ctx.p.x, ctx.tex_ctx.p.y, ctx.tex_ctx.p.z),
            DebugMaterialMode::Uv => Rgb::new(ctx.tex_ctx.uv.x, ctx.tex_ctx.uv.y, 0.0),
            DebugMaterialMode::Wo => Rgb::new(ctx.wo.x, ctx.wo.y, ctx.wo.z),
            DebugMaterialMode::NormalShading => Rgb::new(ctx.ns.x, ctx.ns.y, ctx.ns.z),
            DebugMaterialMode::Dpdus => Rgb::new(ctx.dpdus.x, ctx.dpdus.y, ctx.dpdus.z),
            DebugMaterialMode::Dpdx => Rgb::new(ctx.tex_ctx.dpdx.x, ctx.tex_ctx.dpdx.y, ctx.tex_ctx.dpdx.z),
            DebugMaterialMode::Dpdy => Rgb::new(ctx.tex_ctx.dpdy.x, ctx.tex_ctx.dpdy.y, ctx.tex_ctx.dpdy.z),
            DebugMaterialMode::Dudxy => Rgb::new(ctx.tex_ctx.dudx, ctx.tex_ctx.dudy, 0.0),
            DebugMaterialMode::Dvdxy => Rgb::new(ctx.tex_ctx.dvdx, ctx.tex_ctx.dvdy, 0.0),
            DebugMaterialMode::St => {
                if let SpectrumTexture::Image(imt) = self.color.as_ref() {
                    let mut c = imt.base.mapping.map(&ctx.tex_ctx);
                    c.st[1] = 1.0 - c.st[1];

                    Rgb::new(c.st.x, c.st.y, 0.0)
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            },
            DebugMaterialMode::Texture => {
                if let SpectrumTexture::Image(imt) = self.color.as_ref() {
                    let mut c = imt.base.mapping.map(&ctx.tex_ctx);
                    c.st[1] = 1.0 - c.st[1];

                    let rgb = imt.base.mipmap.filter::<Rgb>(c.st, Vec2f::new(c.dsdx, c.dtdx), Vec2f::new(c.dsdy, c.dtdy)) * imt.base.scale;
                    rgb.clamp_zero()
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            },
        }
    }
}

impl AbstractMaterial for DebugMaterial {
    type ConcreteBxDF = DiffuseBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        DiffuseBxDF::new(SampledSpectrum::from_const(0.0))
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::Diffuse(bxdf))
    }

    fn get_bssrdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Option<BSSRDF> {
        None
    }

    fn can_evaluate_textures<T: AbstractTextureEvaluator>(&self, tex_eval: &T) -> bool {
        tex_eval.can_evaluate(&[], &[Some(&self.color)])
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}
