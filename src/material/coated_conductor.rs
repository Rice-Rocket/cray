use std::{collections::HashMap, sync::Arc};

use crate::{bsdf::BSDF, bssrdf::BSSRDF, bxdf::{conductor::ConductorBxDF, dielectric::DielectricBxDF, layered::CoatedConductorBxDF, BxDF}, color::{named_spectrum::NamedSpectrum, sampled::SampledSpectrum, spectrum::{AbstractSpectrum, ConstantSpectrum, Spectrum}, wavelengths::SampledWavelengths}, error, image::Image, reader::{error::ParseResult, paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary}, target::FileLoc}, scattering::TrowbridgeReitzDistribution, texture::{FloatTexture, SpectrumConstantTexture, SpectrumTexture}, Float};

use super::{AbstractMaterial, AbstractTextureEvaluator, MaterialEvalContext};

#[derive(Debug, Clone)]
pub struct CoatedConductorMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    interface_u_roughness: Arc<FloatTexture>,
    interface_v_roughness: Arc<FloatTexture>,
    thickness: Arc<FloatTexture>,
    interface_eta: Arc<Spectrum>,
    g: Arc<FloatTexture>,
    albedo: Arc<SpectrumTexture>,
    conductor_u_roughness: Arc<FloatTexture>,
    conductor_v_roughness: Arc<FloatTexture>,
    conductor_eta: Option<Arc<SpectrumTexture>>,
    k: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    remap_roughness: bool,
    max_depth: i32,
    n_samples: i32,
}

impl CoatedConductorMaterial {
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        interface_u_roughness: Arc<FloatTexture>,
        interface_v_roughness: Arc<FloatTexture>,
        thickness: Arc<FloatTexture>,
        interface_eta: Arc<Spectrum>,
        g: Arc<FloatTexture>,
        albedo: Arc<SpectrumTexture>,
        conductor_u_roughness: Arc<FloatTexture>,
        conductor_v_roughness: Arc<FloatTexture>,
        conductor_eta: Option<Arc<SpectrumTexture>>,
        k: Option<Arc<SpectrumTexture>>,
        reflectance: Option<Arc<SpectrumTexture>>,
        remap_roughness: bool,
        max_depth: i32,
        n_samples: i32,
    ) -> CoatedConductorMaterial {
        CoatedConductorMaterial {
            displacement,
            normal_map,
            interface_u_roughness,
            interface_v_roughness,
            thickness,
            interface_eta,
            g,
            albedo,
            conductor_u_roughness,
            conductor_v_roughness,
            conductor_eta,
            k,
            reflectance,
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
    ) -> ParseResult<CoatedConductorMaterial> {
        let interface_u_roughness = if let Some(u_rough) = parameters.get_float_texture_or_none("interface.uroughness", textures)? {
            u_rough
        } else {
            parameters.get_float_texture("interface.roughness", 0.0, textures)?
        };
        let interface_v_roughness = if let Some(v_rough) = parameters.get_float_texture_or_none("interface.vroughness", textures)? {
            v_rough
        } else {
            parameters.get_float_texture("interface.roughness", 0.0, textures)?
        };

        let thickness = parameters.get_float_texture("thickness", 0.01, textures)?;

        let interface_eta = if !parameters.get_float_array("interface.eta")?.is_empty() {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(parameters.get_float_array("interface.eta")?[0]))))
        } else {
            parameters.get_one_spectrum("interface.eta", None, SpectrumType::Unbounded, cached_spectra)?
        };

        let interface_eta = if let Some(eta) = interface_eta {
            eta
        } else {
            Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5)))
        };

        let conductor_u_roughness = if let Some(u_rough) = parameters.get_float_texture_or_none("conductor.uroughness", textures)? {
            u_rough
        } else {
            parameters.get_float_texture("conductor.roughness", 0.0, textures)?
        };
        let conductor_v_roughness = if let Some(v_rough) = parameters.get_float_texture_or_none("conductor.vroughness", textures)? {
            v_rough
        } else {
            parameters.get_float_texture("conductor.roughness", 0.0, textures)?
        };

        let mut conductor_eta = parameters.get_spectrum_texture_or_none("conductor.eta", SpectrumType::Unbounded, cached_spectra, textures)?;
        let mut k = parameters.get_spectrum_texture("k", None, SpectrumType::Unbounded, cached_spectra, textures)?;
        let reflectance = parameters.get_spectrum_texture("reflectance", None, SpectrumType::Albedo, cached_spectra, textures)?;

        if reflectance.is_some() && (conductor_eta.is_some() || k.is_some()) {
            error!(loc, ValueConflict, "cannot specify both reflectance and conductor eta/k for conductor material");
        }

        if reflectance.is_none() && conductor_eta.is_none() {
            conductor_eta = Some(Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Spectrum::get_named_spectrum(NamedSpectrum::CuEta)))))
        }

        if reflectance.is_none() && k.is_none() {
            k = Some(Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Spectrum::get_named_spectrum(NamedSpectrum::CuK)))))
        }

        let max_depth = parameters.get_one_int("maxdepth", 10)?;
        let n_samples = parameters.get_one_int("nsamples", 1)?;

        if conductor_eta.is_some() {
            assert!(k.is_some());
        } else {
            assert!(reflectance.is_some());
        }

        let g = parameters.get_float_texture("g", 0.0, textures)?;
        let albedo = if let Some(albedo) = parameters.get_spectrum_texture("albedo", None, SpectrumType::Albedo, cached_spectra, textures)? {
            albedo
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.0))))))
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures)?;
        let remap_roughness = parameters.get_one_bool("remaproughness", true)?;

        Ok(CoatedConductorMaterial {
            displacement,
            normal_map,
            interface_u_roughness,
            interface_v_roughness,
            thickness,
            interface_eta,
            g,
            albedo,
            conductor_u_roughness,
            conductor_v_roughness,
            conductor_eta,
            k,
            reflectance,
            remap_roughness,
            max_depth,
            n_samples,
        })
    }
}

impl AbstractMaterial for CoatedConductorMaterial {
    type ConcreteBxDF = CoatedConductorBxDF;

    fn get_bxdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut iurough = tex_eval.evaluate_float(&self.interface_u_roughness, &ctx.tex_ctx);
        let mut ivrough = tex_eval.evaluate_float(&self.interface_v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            iurough = TrowbridgeReitzDistribution::roughness_to_alpha(iurough);
            ivrough = TrowbridgeReitzDistribution::roughness_to_alpha(ivrough);
        }

        let interface_distribution = TrowbridgeReitzDistribution::new(iurough, ivrough);
        let thick = tex_eval.evaluate_float(&self.thickness, &ctx.tex_ctx);

        let mut ieta = self.interface_eta.get(lambda[0]);
        match self.interface_eta.as_ref() {
            Spectrum::Constant(_) => {},
            _ => lambda.terminate_secondary()
        }

        if ieta == 0.0 {
            ieta = 1.0;
        }

        let (mut ce, mut ck) = if let Some(conductor_eta) = &self.conductor_eta {
            let k = self.k.as_ref().unwrap();
            let ce = tex_eval.evaluate_spectrum(conductor_eta, &ctx.tex_ctx, lambda);
            let ck = tex_eval.evaluate_spectrum(k, &ctx.tex_ctx, lambda);
            (ce, ck)
        } else {
            let reflectance = self.reflectance.as_ref().unwrap();
            let r = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(reflectance, &ctx.tex_ctx, lambda), 0.0, 0.9999);
            let ce = SampledSpectrum::from_const(1.0);
            let ck = 2.0 * r.sqrt() / SampledSpectrum::clamp_zero(&(SampledSpectrum::from_const(1.0) - r)).sqrt();
            (ce, ck)
        };

        ce /= ieta;
        ck /= ieta;

        let mut curough = tex_eval.evaluate_float(&self.conductor_u_roughness, &ctx.tex_ctx);
        let mut cvrough = tex_eval.evaluate_float(&self.conductor_v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            curough = TrowbridgeReitzDistribution::roughness_to_alpha(iurough);
            cvrough = TrowbridgeReitzDistribution::roughness_to_alpha(ivrough);
        }

        let conductor_distrib = TrowbridgeReitzDistribution::new(curough, cvrough);

        let a = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(&self.albedo, &ctx.tex_ctx, lambda), 0.0, 1.0);
        let g = Float::clamp(tex_eval.evaluate_float(&self.g, &ctx.tex_ctx), -1.0, 1.0);

        CoatedConductorBxDF::new(
            DielectricBxDF::new(ieta, interface_distribution),
            ConductorBxDF::new(conductor_distrib, ce, ck),
            thick,
            a,
            g,
            self.max_depth,
            self.n_samples,
        )
    }

    fn get_bsdf<T: AbstractTextureEvaluator>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, BxDF::CoatedConductor(bxdf))
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
            &[
                Some(&self.interface_u_roughness),
                Some(&self.interface_v_roughness),
                Some(&self.thickness),
                Some(&self.g),
                Some(&self.conductor_u_roughness),
                Some(&self.conductor_v_roughness),
            ],
            &[Some(&self.albedo), self.reflectance.as_ref()]
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
