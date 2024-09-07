use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{dielectric::DielectricBxDF, BxDF}, color::{spectrum::{AbstractSpectrum, ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, scattering::TrowbridgeReitzDistribution, texture::FloatTexture};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct DielectricMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
    eta: Arc<Spectrum>,
}

impl DielectricMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> DielectricMaterial {
        let eta = if !parameters.get_float_array("eta").is_empty() {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(
                parameters.get_float_array("eta")[0],
            ))))
        } else {
            parameters.get_one_spectrum("eta", None, SpectrumType::Unbounded, cached_spectra)
        }.unwrap_or(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5))));

        let u_roughness = if let Some(roughness) = parameters.get_float_texture_or_none("uroughness", textures) {
            roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)
        };
        let v_roughness = if let Some(roughness) = parameters.get_float_texture_or_none("vroughness", textures) {
            roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let remap_roughness = parameters.get_one_bool("remaproughness", true);

        DielectricMaterial::new(
            displacement,
            normal_map,
            u_roughness,
            v_roughness,
            remap_roughness,
            eta,
        )
    }

    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        remap_roughness: bool,
        eta: Arc<Spectrum>
    ) -> DielectricMaterial {
        DielectricMaterial {
            displacement,
            normal_map,
            u_roughness,
            v_roughness,
            remap_roughness,
            eta,
        }
    }
}

impl AbstractMaterial for DielectricMaterial {
    type ConcreteBxDF = DielectricBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut sampled_eta = self.eta.get(lambda[0]);
        let is_eta_constant = matches!(self.eta.as_ref(), Spectrum::Constant(_));

        if !is_eta_constant {
            lambda.terminate_secondary();
        }

        if sampled_eta == 0.0 {
            sampled_eta = 1.0;
        }

        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }

        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);
        DielectricBxDF::new(sampled_eta, distrib)
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::Dielectric(bxdf))
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
        tex_eval.can_evaluate(&[Some(self.u_roughness.clone()), Some(self.v_roughness.clone())], &[])
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
