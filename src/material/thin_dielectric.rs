use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{thin_dielectric::ThinDielectricBxDF, BxDF}, color::{spectrum::{AbstractSpectrum, ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, texture::FloatTexture};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct ThinDielectricMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    eta: Arc<Spectrum>,
}

impl ThinDielectricMaterial {
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        eta: Arc<Spectrum>,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            eta,
        }
    }

    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ThinDielectricMaterial {
        let eta = if !parameters.get_float_array("eta").is_empty() {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(parameters.get_float_array("eta")[0]))))
        } else {
            parameters.get_one_spectrum("eta", None, SpectrumType::Unbounded, cached_spectra)
        };

        let eta = if let Some(eta) = eta {
            eta
        } else {
            Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5)))
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        ThinDielectricMaterial::new(displacement, normal_map, eta)
    }
}

impl AbstractMaterial for ThinDielectricMaterial {
    type ConcreteBxDF = ThinDielectricBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut sampled_eta = self.eta.get(lambda[0]);
        match self.eta.as_ref() {
            Spectrum::Constant(_) => {},
            _ => lambda.terminate_secondary(),
        }
        
        if sampled_eta == 0.0 {
            sampled_eta = 1.0;
        }

        ThinDielectricBxDF::new(sampled_eta)
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::ThinDielectric(bxdf))
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
        true
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
