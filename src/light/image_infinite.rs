use std::{io::Write, sync::Arc};

use num::Zero;

use crate::{clear_log, color::{colorspace::RgbColorSpace, rgb_xyz::Rgb, sampled::SampledSpectrum, spectrum::{AbstractSpectrum, RgbIlluminantSpectrum}, wavelengths::SampledWavelengths}, equal_area_sphere_to_square, equal_area_square_to_sphere, image::{Image, WrapMode}, interaction::Interaction, log, reader::utils::truncate_filename, sampling::PiecewiseConstant2D, sqr, transform::{ApplyInverseTransform, ApplyTransform, Transform}, Bounds2f, Bounds3f, Float, Normal3f, Point2f, Point2i, Point3f, Ray, Vec3f, PI};

use super::{AbstractLight, LightBase, LightBounds, LightLiSample, LightSampleContext, LightType};

#[derive(Debug, Clone)]
pub struct ImageInfiniteLight {
    base: LightBase,
    image: Arc<Image>,
    image_color_space: Arc<RgbColorSpace>,
    scale: Float,
    scene_center: Point3f,
    scene_radius: Float,
    distribution: PiecewiseConstant2D,
    compensated_distribution: PiecewiseConstant2D,
}

impl AbstractLight for ImageInfiniteLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut sum_l = SampledSpectrum::from_const(0.0);

        let width = self.image.resolution().x;
        let height = self.image.resolution().y;
        for v in 0..height
        {
            for u in 0..width
            {
                let mut rgb = Rgb::default();
                for c in  0..3 
                {
                    rgb[c] = self.image.get_channel_wrapped(Point2i::new(u,v), c, WrapMode::OctahedralSphere.into());
                }
                sum_l += RgbIlluminantSpectrum::new(&self.image_color_space, &rgb.clamp_zero()).sample(lambda);
            }
        }

        4.0 * PI * PI * sqr(self.scene_radius) * self.scale * sum_l / (width * height) as Float
    }

    fn light_type(&self) -> LightType {
        self.base.ty
    }

    fn sample_li(
        &self,
        ctx: &LightSampleContext,
        u: Point2f,
        lambda: &SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let (uv, map_pdf, _offset) = if allow_incomplete_pdf
        {
            self.compensated_distribution.sample(u)
        } else {
            self.distribution.sample(u)
        };
        if map_pdf == 0.0 
        {
            return None;
        }

        let w_light = equal_area_square_to_sphere(uv.into());
        let wi = self.base.render_from_light.apply(w_light);

        let pdf = map_pdf / (4.0 * PI);

        // TODO: We'll want to include the medium interface here, when we add media.
        // I also want to rework Interaction ctors.
        let intr = Interaction::new(
            (ctx.p() + wi * (2.0 * self.scene_radius)).into(),
            Default::default(),
            Default::default(),
            Default::default(),
            Default::default(),
        );

        Some(LightLiSample::new(
            self.image_le(uv, lambda), wi, pdf, intr))
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        let w_light = self.base.render_from_light.apply_inverse(wi);
        let uv = equal_area_sphere_to_square(w_light);
        let pdf = if allow_incomplete_pdf
        {
            self.compensated_distribution.pdf(uv)
        } else {
            self.distribution.pdf(uv)
        };
        pdf / (4.0 * PI)
    }

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vec3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        let w_light = self.base.render_from_light.apply_inverse(ray.direction);
        let uv = equal_area_sphere_to_square(w_light);
        self.image_le(uv, lambda)
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        let sphere = scene_bounds.bounding_sphere();
        self.scene_center = sphere.0;
        self.scene_radius = sphere.1;
    }

    fn bounds(&self) -> Option<LightBounds> {
        todo!()
    }
}

impl ImageInfiniteLight {
    pub fn new(
        render_from_light: Transform,
        image: Arc<Image>,
        image_color_space: Arc<RgbColorSpace>,
        scale: Float,
        filename: &str,
    ) -> ImageInfiniteLight {
        let base = LightBase{
            ty: LightType::Infinite,
            render_from_light,
            medium: None,
        };

        log!("Creating image infinite light from file '{}'...", truncate_filename(filename));

        let channel_desc = image.get_channel_desc(&["R", "G", "B"]);
        if channel_desc.is_none(){
            panic!("{} Image used for ImageInfiniteLight doesn't have RGB channels", filename);
        }
        let channel_desc = channel_desc.unwrap();
        assert!(channel_desc.size() == 3);
        assert!(channel_desc.is_identity());
        if image.resolution().x != image.resolution().y
        {
            panic!("{} Image resolution is non-square; it is unlikely that it is an environment map", filename);
        }

        let mut d = image.get_default_sampling_distribution();
        let domain = Bounds2f::new(Point2f::ZERO, Point2f::ONE);
        let distribution = PiecewiseConstant2D::new_from_2d(&d, domain);

        let average: Float = d.data.iter().sum::<Float>() / d.data.len() as Float;
        d.data.iter_mut().for_each(|v| *v = Float::max(*v - average, 0.0));
        if d.data.iter().all(|v| v.is_zero())
        {
            d.data.iter_mut().for_each(|v| *v = 1.0 );
        }
        let compensated_distribution = PiecewiseConstant2D::new_from_2d(&d, domain);

        clear_log!();

        ImageInfiniteLight
        {
            base,
            image,
            image_color_space,
            scale,
            scene_center: Default::default(),
            scene_radius: 0.0,
            distribution,
            compensated_distribution,
        }
    }

    fn image_le(&self, uv: Point2f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut rgb = Rgb::default();
        for c in 0..3
        {
            rgb[c] = self.image.lookup_nearest_channel_wrapped(uv, c, WrapMode::OctahedralSphere.into());
        }
        let spec = RgbIlluminantSpectrum::new(&self.image_color_space, &rgb.clamp_zero());
        self.scale * spec.sample(lambda)
    }
}
