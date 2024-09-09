use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{diffuse_transmission::DiffuseTransmissionBxDF, BxDF}, color::{spectrum::{ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}, Float};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct DiffuseTransmissionMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    reflectance: Arc<SpectrumTexture>,
    transmission: Arc<SpectrumTexture>,
    scale: Float,
}

impl DiffuseTransmissionMaterial {
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        reflectance: Arc<SpectrumTexture>,
        transmission: Arc<SpectrumTexture>,
        scale: Float,
    ) -> DiffuseTransmissionMaterial {
        DiffuseTransmissionMaterial {
            displacement,
            normal_map,
            reflectance,
            transmission,
            scale,
        }
    }

    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> DiffuseTransmissionMaterial {
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
                Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.25))),
            )))
        };

        let transmittance = parameters.get_spectrum_texture(
            "transmittance",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        );

        let transmittance = if let Some(transmittance) = transmittance {
            transmittance
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(
                Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.25))),
            )))
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let scale = parameters.get_one_float("scale", 1.0);

        DiffuseTransmissionMaterial::new(
            displacement,
            normal_map,
            reflectance,
            transmittance,
            scale,
        )
    }
}

impl AbstractMaterial for DiffuseTransmissionMaterial {
    type ConcreteBxDF = DiffuseTransmissionBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let r = (self.scale * tex_eval.evaluate_spectrum(&self.reflectance, &ctx.tex_ctx, lambda)).clamp(0.0, 1.0);
        let t = (self.scale * tex_eval.evaluate_spectrum(&self.transmission, &ctx.tex_ctx, lambda)).clamp(0.0, 1.0);
        DiffuseTransmissionBxDF::new(r, t)
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::DiffuseTransmission(bxdf))
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
        tex_eval.can_evaluate(&[], &[Some(self.reflectance.clone()), Some(self.transmission.clone())])
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
