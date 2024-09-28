use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{conductor::ConductorBxDF, BxDF}, color::{named_spectrum::NamedSpectrum, sampled::SampledSpectrum, spectrum::Spectrum, wavelengths::SampledWavelengths}, image::Image, reader::{error::ParseResult, paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, scattering::TrowbridgeReitzDistribution, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct ConductorMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    eta: Option<Arc<SpectrumTexture>>,
    k: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
}

impl ConductorMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ParseResult<ConductorMaterial> {
        let mut eta = parameters.get_spectrum_texture_or_none(
            "eta",
            SpectrumType::Unbounded,
            cached_spectra,
            textures,
        )?;
        let mut k = parameters.get_spectrum_texture(
            "k",
            None,
            SpectrumType::Unbounded,
            cached_spectra,
            textures,
        )?;
        let reflectance = parameters.get_spectrum_texture(
            "reflectance",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        )?;

        if reflectance.is_none() {
            if eta.is_none() {
                eta = Some(Arc::new(SpectrumTexture::Constant(
                    SpectrumConstantTexture::new(Spectrum::get_named_spectrum(
                        NamedSpectrum::CuEta,
                    ))
                )));
            }
            if k.is_none() {
                k = Some(Arc::new(SpectrumTexture::Constant(
                    SpectrumConstantTexture::new(Spectrum::get_named_spectrum(NamedSpectrum::CuK))
                )));
            }
        }

        let u_roughness = if let Some(roughness) = parameters.get_float_texture_or_none("uroughness", textures)? {
            roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)?
        };
        let v_roughness = if let Some(roughness) = parameters.get_float_texture_or_none("vroughness", textures)? {
            roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)?
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures)?;
        let remap_roughness = parameters.get_one_bool("remaproughness", true)?;

        Ok(ConductorMaterial::new(
            displacement,
            normal_map,
            eta,
            k,
            reflectance,
            u_roughness,
            v_roughness,
            remap_roughness,
        ))
    }

    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        eta: Option<Arc<SpectrumTexture>>,
        k: Option<Arc<SpectrumTexture>>,
        reflectance: Option<Arc<SpectrumTexture>>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        remap_roughness: bool,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            eta,
            k,
            reflectance,
            u_roughness,
            v_roughness,
            remap_roughness,
        }
    }
}

impl AbstractMaterial for ConductorMaterial {
    type ConcreteBxDF = ConductorBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }

        let (etas, ks) = if let Some(eta) = &self.eta {
            let k = self.k.as_ref().expect("eta and k should be provided together");
            let etas = tex_eval.evaluate_spectrum(eta, &ctx.tex_ctx, lambda);
            let ks = tex_eval.evaluate_spectrum(k, &ctx.tex_ctx, lambda);
            (etas, ks)
        } else {
            let r = SampledSpectrum::clamp(
                &tex_eval.evaluate_spectrum(
                    self.reflectance.as_ref().expect("if eta/k not present, reflectance should be provided"),
                    &ctx.tex_ctx,
                    lambda
                ),
                0.0,
                0.0,
            );
            let etas = SampledSpectrum::from_const(1.0);
            let ks = 2.0 * SampledSpectrum::sqrt(&r)
                / SampledSpectrum::sqrt(&SampledSpectrum::clamp_zero(
                    &(SampledSpectrum::from_const(1.0) - r),
                ));
            (etas, ks)
        };

        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);
        ConductorBxDF::new(distrib, etas, ks)
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::Conductor(bxdf))
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
        tex_eval.can_evaluate(
            &[Some(&self.u_roughness), Some(&self.v_roughness)],
            &[self.k.as_ref(), self.eta.as_ref(), self.reflectance.as_ref()],
        )
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
