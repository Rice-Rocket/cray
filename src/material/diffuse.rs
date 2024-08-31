use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{diffuse::DiffuseBxDF, BxDF}, color::{spectrum::{ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct DiffuseMaterial {
    reflectance: Arc<SpectrumTexture>,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
}

impl DiffuseMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        _loc: &FileLoc,
    ) -> DiffuseMaterial {
        let reflectance = parameters.get_spectrum_texture(
            "reflectance",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        );

        let reflectance = if let Some(reflectance) = reflectance {
            reflectance
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(
                Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5))),
            )))
        };

        let displacement = Some(parameters.get_float_texture("displacement", 0.0, textures));

        DiffuseMaterial::new(reflectance, displacement, normal_map)
    }

    pub fn new(
        reflectance: Arc<SpectrumTexture>,
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
    ) -> DiffuseMaterial {
        DiffuseMaterial {
            reflectance,
            displacement,
            normal_map,
        }
    }
}

impl AbstractMaterial for DiffuseMaterial {
    type ConcreteBxDF = DiffuseBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let r = tex_eval
            .evaluate_spectrum(&self.reflectance, &ctx.tex_ctx, lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(r)
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
        tex_eval.can_evaluate(&[], &[Some(self.reflectance.clone())])
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
