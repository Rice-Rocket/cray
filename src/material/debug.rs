use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{diffuse::DiffuseBxDF, BxDF}, color::{rgb_xyz::Rgb, sampled::SampledSpectrum, spectrum::{ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub enum DebugMaterialMode {
    Normal,
    Position,
    UV,
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
        _loc: &FileLoc,
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
            "uv" => DebugMaterialMode::UV,
            s => panic!("unknown debug material mode {}", s),
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
            DebugMaterialMode::UV => Rgb::new(ctx.tex_ctx.uv.x, ctx.tex_ctx.uv.y, 0.0),
        }
    }
}

impl AbstractMaterial for DebugMaterial {
    type ConcreteBxDF = DiffuseBxDF;

    fn get_bxdf<T: super::AbstractTextureEvaluator>(
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
