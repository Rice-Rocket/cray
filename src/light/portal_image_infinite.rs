use std::sync::Arc;

use crate::{color::{colorspace::RgbColorSpace, rgb_xyz::Rgb, sampled::SampledSpectrum, spectrum::{AbstractSpectrum, RgbIlluminantSpectrum}, wavelengths::SampledWavelengths}, equal_area_sphere_to_square, error, image::{Image, PixelFormat, WrapMode}, interaction::Interaction, light::{LightBase, LightType}, reader::{error::ParseResult, target::FileLoc}, sampling::WindowedPiecewiseConstant2D, sqr, transform::{ApplyInverseTransform, Transform}, Bounds2f, Bounds3f, Dot, Float, Frame, Normal3f, Point2f, Point2i, Point3f, Ray, Vec3f, PI};

use super::{AbstractLight, LightBounds, LightLiSample, LightSampleContext};

#[derive(Debug, Clone)]
pub struct PortalImageInfiniteLight {
    base: LightBase,
    portal: [Point3f; 4],
    portal_frame: Frame,
    image: Arc<Image>,
    distribution: WindowedPiecewiseConstant2D,
    image_color_space: Arc<RgbColorSpace>,
    scale: Float,
    scene_radius: Float,
    scene_center: Point3f,
}

impl PortalImageInfiniteLight {
    pub fn new(
        render_from_light: Transform,
        equal_area_image: Arc<Image>,
        image_color_space: Arc<RgbColorSpace>,
        scale: Float,
        filename: &str,
        p: Vec<Point3f>,
        loc: &FileLoc,
    ) -> ParseResult<PortalImageInfiniteLight> {
        let base = LightBase {
            ty: LightType::Infinite,
            render_from_light,
            medium: None,
        };

        let Some(channel_desc) = equal_area_image.get_channel_desc(&["R", "G", "B"]) else {
            error!(@file filename, "image used for PortalImageInfiniteLight doesn't have R, G, B channels");
        };

        assert_eq!(channel_desc.size(), 3);
        assert!(channel_desc.is_identity());

        if equal_area_image.resolution().x != equal_area_image.resolution().y {
            error!(@file filename, "image resolution ({}, {}), is non-square", equal_area_image.resolution().x, equal_area_image.resolution().y);
        }

        if p.len() != 4 {
            error!(loc, "expected 4 vertices for infinite light portal but given {}", p.len());
        }

        let mut portal = [Point3f::ZERO; 4];
        portal.copy_from_slice(&p[..4]);

        let p01 = (portal[1] - portal[0]).normalize();
        let p12 = (portal[2] - portal[1]).normalize();
        let p32 = (portal[2] - portal[3]).normalize();
        let p03 = (portal[3] - portal[0]).normalize();

        if (p01.dot(p32) - 1.0).abs() > 0.001 || (p12.dot(p03) - 1.0) > 0.001 {
            error!(loc, "infinite light portal isn't a planar quadrilateral");
        }

        if p01.dot(p12).abs() > 0.001 || p12.dot(p32).abs() > 0.001
        || p32.dot(p03).abs() > 0.001 || p03.dot(p01).abs() > 0.001 {
            error!(loc, "infinite light portal isn't a planar quadrilateral");
        }

        let portal_frame = Frame::from_xy(p03.into(), p01.into());

        let mut image = Image::new(
            PixelFormat::Float32,
            equal_area_image.resolution(),
            &["R".to_string(), "G".to_string(), "B".to_string()],
            equal_area_image.encoding().cloned(),
        );

        // TODO: parallelize this
        for y in 0..image.resolution().y {
            for x in 0..image.resolution().x {
                let uv = Point2f::new(
                    (x as Float + 0.5) / image.resolution().x as Float,
                    (y as Float + 0.5) / image.resolution().y as Float,
                );
                let mut w = Self::render_from_image(&portal_frame, uv, None);
                w = render_from_light.apply_inverse(w).normalize();
                let uv_equi = equal_area_sphere_to_square(w);

                for c in 0..3 {
                    let v = equal_area_image.bilinear_channel_wrapped(uv_equi, c, WrapMode::OctahedralSphere.into());
                    image.set_channel(Point2i::new(x, y), c, v);
                }
            }
        }

        let duv_dw = |p: Point2f| {
            let mut duv_dw = 0.0;
            Self::render_from_image(&portal_frame, p, Some(&mut duv_dw));
            duv_dw
        };

        let d = image.get_sampling_distribution(duv_dw, &Bounds2f::new(Point2f::ZERO, Point2f::ONE));
        let distribution = WindowedPiecewiseConstant2D::new(d);

        Ok(PortalImageInfiniteLight {
            base,
            portal,
            portal_frame,
            image: Arc::new(image),
            distribution,
            image_color_space,
            scale,
            scene_radius: 0.0,
            scene_center: Default::default(),
        })
    }

    fn render_from_image(portal_frame: &Frame, uv: Point2f, duv_dw: Option<&mut Float>) -> Vec3f {
        let alpha = -PI / 2.0 + uv[0] * PI;
        let beta = -PI / 2.0 + uv[1] * PI;
        let x = alpha.tan();
        let y = beta.tan();
        debug_assert!(!x.is_infinite() && !y.is_infinite());
        let w = Vec3f::new(x, y, 1.0).normalize();

        if let Some(duv_dw) = duv_dw {
            *duv_dw = sqr(PI) * (1.0 - sqr(w.x)) * (1.0 - sqr(w.y)) / w.z;
        }

        portal_frame.from_local(w)
    }

    fn image_from_render(&self, w_render: Vec3f, duv_dw: Option<&mut Float>) -> Option<Point2f> {
        let w = self.portal_frame.localize(w_render);
        if w.z <= 0.0 {
            return None;
        }

        if let Some(duv_dw) = duv_dw {
            *duv_dw = sqr(PI) * (1.0 - sqr(w.x)) * (1.0 - sqr(w.y)) / w.z;
        }

        let alpha = Float::atan2(w.x, w.z);
        let beta = Float::atan2(w.y, w.z);
        debug_assert!(!(alpha + beta).is_nan());

        Some(Point2f::new(
            Float::clamp((alpha + PI / 2.0) / PI, 0.0, 1.0),
            Float::clamp((beta + PI / 2.0) / PI, 0.0, 1.0),
        ))
    }

    fn image_bounds(&self, p: Point3f) -> Option<Bounds2f> {
        let p0 = self.image_from_render((self.portal[0] - p).normalize().into(), None);
        let p1 = self.image_from_render((self.portal[2] - p).normalize().into(), None);

        if let Some(p0) = p0 {
            if let Some(p1) = p1 {
                return Some(Bounds2f::new(p0, p1));
            }
        }

        None
    }

    fn area(&self) -> Float {
        (self.portal[1] - self.portal[0]).length() * (self.portal[3] - self.portal[0]).length()
    }

    fn image_lookup(&self, uv: Point2f, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut rgb = Rgb::new(0.0, 0.0, 0.0);
        for c in 0..3 {
            rgb[c] = self.image.lookup_nearest_channel(uv, c);
        }

        let spec = RgbIlluminantSpectrum::new(self.image_color_space.as_ref(), &rgb.clamp_zero());
        self.scale * spec.sample(lambda)
    }
}

impl AbstractLight for PortalImageInfiniteLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut sum_l = SampledSpectrum::from_const(0.0);

        for y in 0..self.image.resolution().y {
            for x in 0..self.image.resolution().x {
                let mut rgb = Rgb::new(0.0, 0.0, 0.0);
                for c in 0..3 {
                    rgb[c] = self.image.get_channel(Point2i::new(x, y), c);
                }

                let st = Point2f::new(
                    (x as Float + 0.5) / self.image.resolution().x as Float,
                    (y as Float + 0.5) / self.image.resolution().y as Float,
                );

                let mut duv_dw = 0.0;
                Self::render_from_image(&self.portal_frame, st, Some(&mut duv_dw));

                sum_l += RgbIlluminantSpectrum::new(self.image_color_space.as_ref(), &rgb.clamp_zero())
                    .sample(lambda) / duv_dw;
            }
        }

        self.scale * self.area() * sum_l / (self.image.resolution().x * self.image.resolution().y) as Float
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
        let b = self.image_bounds(ctx.p())?;
        let mut map_pdf = 0.0;
        let uv = self.distribution.sample(u, b, &mut map_pdf)?;

        let mut duv_dw = 0.0;
        let wi = Self::render_from_image(&self.portal_frame, uv, Some(&mut duv_dw));
        if duv_dw == 0.0 {
            return None;
        }

        let pdf = map_pdf / duv_dw;
        debug_assert!(!pdf.is_infinite());

        let l = self.image_lookup(uv, lambda);
        let pl = ctx.p() + 2.0 * self.scene_radius * wi;
        Some(LightLiSample::new(l, wi, pdf, Interaction {
            pi: pl.into(),
            medium_interface: self.base.medium.clone(),
            ..Default::default()
        }))
    }

    fn pdf_li(&self, ctx: &LightSampleContext, wi: Vec3f, allow_incomplete_pdf: bool) -> Float {
        let mut duv_dw = 0.0;
        let Some(uv) = self.image_from_render(wi, Some(&mut duv_dw)) else {
            return 0.0;
        };

        if duv_dw == 0.0 {
            return 0.0;
        }

        let Some(b) = self.image_bounds(ctx.p()) else {
            return 0.0;
        };

        let pdf = self.distribution.pdf(uv, b);
        pdf / duv_dw
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
        let uv = self.image_from_render(ray.direction.normalize(), None);
        let b = self.image_bounds(ray.origin);

        if let Some(uv) = uv {
            if let Some(b) = b {
                if b.inside(uv) {
                    return self.image_lookup(uv, lambda);
                }
            }
        }

        SampledSpectrum::from_const(0.0)
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        let (scene_center, scene_radius) = scene_bounds.bounding_sphere();
        self.scene_center = scene_center;
        self.scene_radius = scene_radius;
    }

    fn bounds(&self) -> Option<LightBounds> {
        None
    }
}
