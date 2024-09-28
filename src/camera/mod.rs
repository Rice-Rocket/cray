use std::{ops::Mul, sync::Arc};

use film::{Film, AbstractFilm as _};
use projective::{OrthographicCamera, PerspectiveCamera};
use transform::{ApplyInverseTransform, ApplyRayInverseTransform, ApplyRayTransform, ApplyTransform};
use vect::Dot;

use crate::{color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, error, image::ImageMetadata, math::*, media::Medium, options::{CameraRenderingSpace, Options}, ray::AbstractRay, reader::{error::ParseResult, paramdict::ParameterDictionary, target::FileLoc}, transform::Transform, warn, AuxiliaryRays, Float, Frame, Normal3f, Point2f, Point3f, Ray, RayDifferential, Vec3f};

pub mod projective;
pub mod film;
pub mod filter;

pub trait AbstractCamera {
    fn generate_ray(&self, sample: &CameraSample, lambda: &SampledWavelengths) -> Option<CameraRay>;
    fn generate_ray_differential(&self, sample: &CameraSample, lambda: &SampledWavelengths) -> Option<CameraRayDifferential>;
    fn get_film_mut(&mut self) -> &mut Arc<Film>;
    fn get_film(&self) -> &Arc<Film>;
    fn sample_time(&self, u: Float) -> Float;
    fn init_metadata(&self, metadata: &mut ImageMetadata);
    fn get_camera_transform(&self) -> &CameraTransform;
    fn approximate_dp_dxy(&self, p: Point3f, n: Normal3f, time: Float, samples_per_pixel: i32, options: &Options) -> (Vec3f, Vec3f);
}

#[derive(Debug, Clone)]
pub enum Camera {
    Orthographic(OrthographicCamera),
    Perspective(PerspectiveCamera),
}

impl Camera {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        medium: Option<Arc<Medium>>,
        camera_transform: CameraTransform,
        film: Arc<Film>,
        options: &Options,
        loc: &FileLoc,
    ) -> ParseResult<Camera> {
        Ok(match name {
            "perspective" => Camera::Perspective(PerspectiveCamera::create(
                parameters,
                camera_transform,
                film,
                medium,
                options,
                loc
            )?),
            "orthographic" => Camera::Orthographic(OrthographicCamera::create(
                parameters,
                camera_transform,
                film,
                medium,
                options,
                loc
            )?),
            _ => { error!(loc, "camera type '{}' unknown", name); },
        })
    }
}

impl AbstractCamera for Camera {
    fn generate_ray(&self, sample: &CameraSample, lambda: &SampledWavelengths) -> Option<CameraRay> {
        match self {
            Camera::Orthographic(c) => c.generate_ray(sample, lambda),
            Camera::Perspective(c) => c.generate_ray(sample, lambda),
        }
    }

    fn generate_ray_differential(&self, sample: &CameraSample, lambda: &SampledWavelengths) -> Option<CameraRayDifferential> {
        match self {
            Camera::Orthographic(c) => c.generate_ray_differential(sample, lambda),
            Camera::Perspective(c) => c.generate_ray_differential(sample, lambda),
        }
    }

    fn get_film(&self) -> &Arc<Film> {
        match self {
            Camera::Orthographic(c) => c.get_film(),
            Camera::Perspective(c) => c.get_film(),
        }
    }

    fn get_film_mut(&mut self) -> &mut Arc<Film> {
        match self {
            Camera::Orthographic(c) => c.get_film_mut(),
            Camera::Perspective(c) => c.get_film_mut(),
        }
    }

    fn sample_time(&self, u: Float) -> Float {
        match self {
            Camera::Orthographic(c) => c.sample_time(u),
            Camera::Perspective(c) => c.sample_time(u),
        }
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        match self {
            Camera::Orthographic(c) => c.init_metadata(metadata),
            Camera::Perspective(c) => c.init_metadata(metadata),
        }
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        match self {
            Camera::Orthographic(c) => c.get_camera_transform(),
            Camera::Perspective(c) => c.get_camera_transform(),
        }
    }

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vec3f, Vec3f) {
        match self {
            Camera::Orthographic(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel, options),
            Camera::Perspective(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel, options),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CameraTransform {
    // TODO: Make this AnimatedTransform
    render_from_camera: Transform,
    world_from_render: Transform,
}

impl CameraTransform {
    pub fn new(world_from_camera: Transform, options: &Options) -> CameraTransform {
        let world_from_render = match options.rendering_space {
            CameraRenderingSpace::Camera => world_from_camera,
            CameraRenderingSpace::CameraWorld => {
                let p_camera = world_from_camera.apply(Point3f::ZERO);
                Transform::from_translation(p_camera)
            },
            CameraRenderingSpace::World => Transform::default(),
        };

        let render_from_world = world_from_render.inverse();
        let render_from_camera = render_from_world.apply(world_from_camera);

        CameraTransform { render_from_camera, world_from_render }
    }

    pub fn render_from_camera<T>(&self, v: T) -> T 
    where Transform: ApplyTransform<T> {
        self.render_from_camera.apply(v)
    }

    pub fn camera_from_render<T>(&self, v: T, _time: Float) -> T
    where Transform: ApplyInverseTransform<T> {
        self.render_from_camera.apply_inverse(v)
    }

    pub fn render_from_camera_ray<T>(&self, v: &T) -> T
    where Transform: ApplyRayTransform<T> {
        self.render_from_camera.apply_ray(v, None)
    }

    pub fn camera_from_render_ray<T>(&self, v: &T) -> T
    where Transform: ApplyRayInverseTransform<T> {
        self.render_from_camera.apply_ray_inverse(v, None)
    }

    pub fn render_from_world_p(&self, p: Point3f) -> Point3f {
        self.world_from_render.apply_inverse(p)
    }

    pub fn render_from_world(&self) -> Transform {
        self.world_from_render.inverse()
    }

    pub fn camera_from_render_mat(&self, _time: Float) -> Transform {
        self.render_from_camera.inverse()
    }

    pub fn camera_from_world(&self, _time: Float) -> Transform {
        (self.world_from_render.apply(self.render_from_camera)).inverse()
    }

    pub fn render_from_camera_mat(&self) -> Transform {
        self.render_from_camera
    }

    pub fn world_from_render(&self) -> Transform {
        self.world_from_render
    }
}

#[derive(Debug, Clone)]
pub struct CameraBase {
    camera_transform: CameraTransform,
    shutter_open: Float,
    shutter_close: Float,
    film: Arc<Film>,
    medium: Option<Arc<Medium>>,
    min_pos_differential_x: Vec3f,
    min_pos_differential_y: Vec3f,
    min_dir_differential_x: Vec3f,
    min_dir_differential_y: Vec3f,
}

impl CameraBase {
    pub fn get_film(&self) -> &Arc<Film> {
        &self.film
    }

    pub fn get_film_mut(&mut self) -> &mut Arc<Film> {
        &mut self.film
    }

    pub fn init_metadata(&self, metadata: &mut ImageMetadata) {
        metadata.camera_from_world = Some(self.camera_transform.camera_from_world(self.shutter_open).m);
    }

    pub fn render_from_camera<T>(&self, v: T) -> T
    where Transform: ApplyTransform<T> {
        self.camera_transform.render_from_camera(v)
    }

    pub fn camera_from_render<T>(&self, v: T, time: Float) -> T
    where Transform: ApplyInverseTransform<T> {
        self.camera_transform.camera_from_render(v, time)
    }

    pub fn render_from_camera_ray<T>(&self, v: &T) -> T
    where Transform: ApplyRayTransform<T> {
        self.camera_transform.render_from_camera_ray(v)
    }

    pub fn camera_from_render_ray<T>(&self, v: &T) -> T
    where Transform: ApplyRayInverseTransform<T> {
        self.camera_transform.camera_from_render_ray(v)
    }

    pub fn sample_time(&self, u: Float) -> Float {
        lerp(self.shutter_open, self.shutter_close, u)
    }

    pub fn generate_ray_differential<T: AbstractCamera>(
        &self,
        camera: &T,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let cr = camera.generate_ray(sample, lambda)?;

        let rx = [0.05, -0.05]
            .iter()
            .map(|eps| -> (Float, CameraSample) {
                let mut sshift = sample.clone();
                sshift.p_film.x += eps;
                (*eps, sshift)
            })
            .find_map(|(eps, sshift)| -> Option<(Point3f, Vec3f)> {
                camera.generate_ray(&sshift, lambda).map(|rx| (
                    cr.ray.origin + (rx.ray.origin - cr.ray.origin) / eps,
                    cr.ray.direction + (rx.ray.direction - cr.ray.direction) / eps,
                ))
            });

        let ry = [0.05, -0.05]
            .iter()
            .map(|eps| -> (Float, CameraSample) {
                let mut sshift = sample.clone();
                sshift.p_film.y += eps;
                (*eps, sshift)
            })
            .find_map(|(eps, sshift)| -> Option<(Point3f, Vec3f)> {
                camera.generate_ray(&sshift, lambda).map(|ry| (
                    cr.ray.origin + (ry.ray.origin - cr.ray.origin) / eps,
                    cr.ray.direction + (ry.ray.direction - cr.ray.direction) / eps,
                ))
            });

        let aux = if let (Some(rx), Some(ry)) = (rx, ry) {
            Some(AuxiliaryRays::new(rx.0, rx.1, ry.0, ry.1))
        } else {
            None
        };

        let rd = RayDifferential::new(cr.ray, aux);

        Some(CameraRayDifferential::new_with_weight(
            rd,
            cr.weight,
        ))
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options
    ) -> (Vec3f, Vec3f) {
        let p_camera = self.camera_from_render(p, time);
        let down_z_from_camera = Transform::from_rotation_delta(p_camera.normalize().into(), Vec3f::new(0.0, 0.0, 1.0));
        let p_down_z = down_z_from_camera.apply(p_camera);
        let n_down_z = down_z_from_camera.apply(self.camera_from_render(n, time));
        let d = n_down_z.z * p_down_z.z;

        let x_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_x,
            Vec3f::new(0.0, 0.0, 1.0) + self.min_dir_differential_x,
        );
        let tx = -(n_down_z.dot(x_ray.origin) - d) / n_down_z.dot(x_ray.direction);
        
        let y_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_y,
            Vec3f::new(0.0, 0.0, 1.0) + self.min_dir_differential_y,
        );

        let ty = -(n_down_z.dot(y_ray.origin) - d) / n_down_z.dot(y_ray.direction);

        let px = x_ray.at(tx);
        let py = y_ray.at(ty);

        let spp_scale = if options.disable_pixel_jitter {
            1.0
        } else {
            Float::max(0.125, 1.0 / (samples_per_pixel as Float).sqrt())
        };

        let dpdx = spp_scale * self.render_from_camera(down_z_from_camera.apply_inverse(Vec3f::from(px - p_down_z)));
        let dpdy = spp_scale * self.render_from_camera(down_z_from_camera.apply_inverse(Vec3f::from(py - p_down_z)));

        (dpdx, dpdy)
    }

    pub fn find_minimum_differentials<T: AbstractCamera>(&mut self, camera: &T) {
        self.min_pos_differential_x = Vec3f::splat(Float::INFINITY);
        self.min_pos_differential_y = Vec3f::splat(Float::INFINITY);
        self.min_dir_differential_x = Vec3f::splat(Float::INFINITY);
        self.min_dir_differential_y = Vec3f::splat(Float::INFINITY);
        
        let mut sample = CameraSample {
            p_film: Point2f::default(),
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            filter_weight: 1.0,
        };

        let lambda = SampledWavelengths::sample_visible(0.5);

        let n = 512;
        for i in 0..n {
            sample.p_film.x = i as Float / (n - 1) as Float * self.film.full_resolution().x as Float;
            sample.p_film.y = i as Float / (n - 1) as Float * self.film.full_resolution().y as Float;

            let Some(mut crd) = camera.generate_ray_differential(&sample, &lambda) else { continue };

            let ray = &mut crd.ray;

            let dox = self.camera_from_render(Vec3f::from(ray.aux.as_ref().unwrap().rx_origin - ray.ray.origin), ray.ray.time);
            if dox.length_squared() < self.min_pos_differential_x.length_squared() {
                self.min_pos_differential_x = dox;
            }

            let doy = self.camera_from_render(Vec3f::from(ray.aux.as_ref().unwrap().ry_origin - ray.ray.origin), ray.ray.time);
            if doy.length_squared() < self.min_pos_differential_y.length_squared() {
                self.min_pos_differential_y = doy;
            }

            ray.ray.direction = ray.ray.direction.normalize();
            ray.aux.as_mut().unwrap().rx_direction = ray.aux.as_ref().unwrap().rx_direction.normalize();
            ray.aux.as_mut().unwrap().ry_direction = ray.aux.as_ref().unwrap().ry_direction.normalize();

            let f = Frame::from_z(ray.ray.direction);
            let df = f.localize(ray.ray.direction);
            let dxf = f.localize(ray.aux.as_ref().unwrap().rx_direction).normalize();
            let dyf = f.localize(ray.aux.as_ref().unwrap().ry_direction).normalize();

            if (dxf - df).length_squared() < self.min_dir_differential_x.length_squared() {
                self.min_dir_differential_x = dxf - df;
            }

            if (dyf - df).length_squared() < self.min_dir_differential_y.length_squared() {
                self.min_dir_differential_y = dyf - df;
            }
        }
    }
}

pub struct CameraBaseParameters {
    pub camera_transform: CameraTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: Option<Arc<Medium>>,
}

impl CameraBaseParameters {
    pub fn new(
        camera_transform: CameraTransform,
        film: Arc<Film>,
        medium: Option<Arc<Medium>>,
        parameters: &mut ParameterDictionary,
        loc: &FileLoc,
    ) -> ParseResult<CameraBaseParameters> {
        let mut shutter_open = parameters.get_one_float("shutteropen", 0.0)?;
        let mut shutter_close = parameters.get_one_float("shutterclose", 1.0)?;

        if shutter_close < shutter_open {
            warn!(loc, "shutter close time [{}] < shutter open time [{}]. swapping them.", shutter_close, shutter_open);
            std::mem::swap(&mut shutter_close, &mut shutter_open);
        }

        Ok(CameraBaseParameters {
            camera_transform,
            shutter_open,
            shutter_close,
            film,
            medium,
        })
    }
}

#[derive(Clone)]
pub struct CameraSample {
    pub p_film: Point2f,
    pub p_lens: Point2f,
    pub time: Float,
    pub filter_weight: Float,
}

impl Default for CameraSample {
    fn default() -> Self {
        Self {
            p_film: Point2f::default(),
            p_lens: Point2f::default(),
            time: 0.0,
            filter_weight: 1.0,
        }
    }
}

pub struct CameraRay {
    pub ray: Ray,
    pub weight: SampledSpectrum,
}

impl CameraRay {
    pub fn new(ray: Ray) -> CameraRay {
        CameraRay {
            ray,
            weight: SampledSpectrum::from_const(1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraRayDifferential {
    pub ray: RayDifferential,
    pub weight: SampledSpectrum,
}

impl CameraRayDifferential {
    pub fn new(ray: RayDifferential) -> CameraRayDifferential {
        CameraRayDifferential {
            ray,
            weight: SampledSpectrum::from_const(1.0),
        }
    }

    pub fn new_with_weight(ray: RayDifferential, weight: SampledSpectrum) -> CameraRayDifferential {
        CameraRayDifferential {
            ray,
            weight,
        }
    }
}
