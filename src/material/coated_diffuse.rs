use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{dielectric::DielectricBxDF, diffuse::DiffuseBxDF, layered::CoatedDiffuseBxDF, BxDF}, color::{sampled::SampledSpectrum, spectrum::{AbstractSpectrum, ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, image::Image, reader::{paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, scattering::TrowbridgeReitzDistribution, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}, Float};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct CoatedDiffuseMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    reflectance: Arc<SpectrumTexture>,
    albedo: Arc<SpectrumTexture>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    thickness: Arc<FloatTexture>,
    g: Arc<FloatTexture>,
    eta: Arc<Spectrum>,
    remap_roughness: bool,
    max_depth: i32,
    n_samples: i32,
}

impl CoatedDiffuseMaterial {
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        reflectance: Arc<SpectrumTexture>,
        albedo: Arc<SpectrumTexture>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        thickness: Arc<FloatTexture>,
        g: Arc<FloatTexture>,
        eta: Arc<Spectrum>,
        remap_roughness: bool,
        max_depth: i32,
        n_samples: i32,
    ) -> CoatedDiffuseMaterial {
        CoatedDiffuseMaterial {
            displacement,
            normal_map,
            reflectance,
            albedo,
            u_roughness,
            v_roughness,
            thickness,
            g,
            eta,
            remap_roughness,
            max_depth,
            n_samples,
        }
    }

    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> CoatedDiffuseMaterial {
        let reflectance = if let Some(reflectance) = parameters.get_spectrum_texture("reflectance", None, SpectrumType::Albedo, cached_spectra, textures) {
            reflectance
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5))))))
        };

        let u_roughness = if let Some(u_roughness) = parameters.get_float_texture_or_none("uroughness", textures) {
            u_roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)
        };
        let v_roughness = if let Some(v_roughness) = parameters.get_float_texture_or_none("vroughness", textures) {
            v_roughness
        } else {
            parameters.get_float_texture("roughness", 0.0, textures)
        };

        let thickness = parameters.get_float_texture("thickness", 0.01, textures);

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

        let max_depth = parameters.get_one_int("maxdepth", 10);
        let n_samples = parameters.get_one_int("nsamples", 1);

        let g = parameters.get_float_texture("g", 0.0, textures);
        let albedo = if let Some(albedo) = parameters.get_spectrum_texture("albedo", None, SpectrumType::Albedo, cached_spectra, textures) {
            albedo
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.0))))))
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let remap_roughness = parameters.get_one_bool("remaproughness", true);

        CoatedDiffuseMaterial {
            displacement,
            normal_map,
            reflectance,
            albedo,
            u_roughness,
            v_roughness,
            thickness,
            g,
            eta,
            remap_roughness,
            max_depth,
            n_samples,
        }
    }
}

impl AbstractMaterial for CoatedDiffuseMaterial {
    type ConcreteBxDF = CoatedDiffuseBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let r = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(&self.reflectance, &ctx.tex_ctx, lambda), 0.0, 1.0);

        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }

        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);
        let thick = tex_eval.evaluate_float(&self.thickness, &ctx.tex_ctx);

        let mut sampled_eta = self.eta.get(lambda[0]);
        match self.eta.as_ref() {
            Spectrum::Constant(_) => {},
            _ => lambda.terminate_secondary()
        }

        if sampled_eta == 0.0 {
            sampled_eta = 1.0;
        }

        let a = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(&self.albedo, &ctx.tex_ctx, lambda), 0.0, 1.0);
        let g = Float::clamp(tex_eval.evaluate_float(&self.g, &ctx.tex_ctx), -1.0, 1.0);

        CoatedDiffuseBxDF::new(
            DielectricBxDF::new(
                sampled_eta,
                distrib,
            ),
            DiffuseBxDF::new(r),
            thick,
            a,
            g,
            self.max_depth,
            self.n_samples
        )
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::CoatedDiffuse(bxdf))
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
            &[Some(self.u_roughness.clone()), Some(self.v_roughness.clone()), Some(self.thickness.clone()), Some(self.g.clone())],
            &[Some(self.reflectance.clone()), Some(self.albedo.clone())],
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
