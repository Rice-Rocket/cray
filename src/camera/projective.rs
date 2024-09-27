use std::sync::Arc;

use crate::{color::{sampled::SampledSpectrum, wavelengths::SampledWavelengths}, image::ImageMetadata, media::Medium, options::Options, ray::AbstractRay, reader::{paramdict::ParameterDictionary, target::FileLoc}, sampling::sample_uniform_disk_concentric, transform::{ApplyTransform, Transform}, warn, AuxiliaryRays, Bounds2f, Float, Normal3f, Point2f, Point3f, Ray, RayDifferential, Vec3f};

use super::{film::{Film, AbstractFilm as _}, filter::AbstractFilter as _, AbstractCamera, CameraBase, CameraBaseParameters, CameraRay, CameraRayDifferential, CameraSample, CameraTransform};

#[derive(Debug, Clone)]
pub struct ProjectiveCamera {
    camera_base: CameraBase,
    screen_from_camera: Transform,
    camera_from_raster: Transform,
    raster_from_screen: Transform,
    screen_from_raster: Transform,
    lens_radius: Float,
    focal_distance: Float,
}

impl ProjectiveCamera {
    pub fn new(
        base_parameters: CameraBaseParameters,
        lens_radius: Float,
        focal_distance: Float,
        screen_from_camera: Transform,
        screen_window: Bounds2f,
    ) -> ProjectiveCamera {
        let camera_base = CameraBase {
            camera_transform: base_parameters.camera_transform,
            shutter_open: base_parameters.shutter_open,
            shutter_close: base_parameters.shutter_close,
            film: base_parameters.film,
            medium: base_parameters.medium,
            min_pos_differential_x: Vec3f::default(),
            min_pos_differential_y: Vec3f::default(),
            min_dir_differential_x: Vec3f::default(),
            min_dir_differential_y: Vec3f::default(),
        };

        let ndc_from_screen = Transform::from_scale(Vec3f::new(
            1.0 / (screen_window.max.x - screen_window.min.x),
            1.0 / (screen_window.max.y - screen_window.min.y),
            1.0
        )).apply(Transform::from_translation(Point3f::new(-screen_window.min.x, -screen_window.max.y, 0.0)));

        let raster_from_ndc = Transform::from_scale(Vec3f::new(
            camera_base.film.full_resolution().x as Float,
            -camera_base.film.full_resolution().y as Float,
            1.0
        ));

        let raster_from_screen = raster_from_ndc.apply(ndc_from_screen);
        let screen_from_raster = raster_from_screen.inverse();
        let camera_from_raster = screen_from_camera.inverse().apply(screen_from_raster);

        ProjectiveCamera {
            camera_base,
            screen_from_camera,
            camera_from_raster,
            raster_from_screen,
            screen_from_raster,
            lens_radius,
            focal_distance,
        }
    }

    pub fn init_metadata(&self, metadata: &mut ImageMetadata) {
        self.camera_base.init_metadata(metadata);
        if let Some(camera_from_world) = metadata.camera_from_world {
            metadata.ndc_from_world = Some(
                Transform::from_translation(Point3f::new(0.5, 0.5, 0.5)).m
                    * Transform::from_scale(Vec3f::new(0.5, 0.5, 0.5)).m
                    * self.screen_from_camera.m
                    * camera_from_world
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrthographicCamera {
    projective: ProjectiveCamera,
    dx_camera: Vec3f,
    dy_camera: Vec3f,
}

impl OrthographicCamera {
    pub fn create(
        parameters: &mut ParameterDictionary,
        camera_transform: CameraTransform,
        film: Arc<Film>,
        medium: Option<Arc<Medium>>,
        options: &Options,
        loc: &FileLoc
    ) -> OrthographicCamera {
        let camera_base_params = CameraBaseParameters::new(camera_transform, film, medium, parameters, loc);

        let lens_radius = parameters.get_one_float("lensradius", 0.0);
        let focal_distance = parameters.get_one_float("focaldistance", 1e6);

        let x = camera_base_params.film.full_resolution().x as Float;
        let y = camera_base_params.film.full_resolution().y as Float;
        let frame = parameters.get_one_float("frameaspectratio", x / y);

        let mut screen = if frame > 1.0 {
            Bounds2f::new(Point2f::new(-frame, -1.0), Point2f::new(frame, 1.0))
        } else {
            Bounds2f::new(Point2f::new(-1.0, -1.0 / frame), Point2f::new(1.0, 1.0 / frame))
        };

        let sw = parameters.get_float_array("screenwindow");

        if !sw.is_empty() {
            if options.fullscreen {
                warn!(loc, "screenwindow is ignored in fullscreen mode");
            } else if sw.len() == 4 {
                screen = Bounds2f::new(Point2f::new(sw[0], sw[2]), Point2f::new(sw[1], sw[3]));
            } else {
                warn!(
                    loc,
                    "expected four values for 'screenwindow' parameter but got {}.",
                    sw.len(),
                );
            }
        }

        OrthographicCamera::new(camera_base_params, lens_radius, focal_distance, screen)
    }

    pub fn new(
        camera_base_params: CameraBaseParameters,
        lens_radius: Float,
        focal_distance: Float,
        screen_window: Bounds2f,
    ) -> OrthographicCamera {
        let screen_from_camera = Transform::orthographic(0.0, 1.0);
        
        let mut projective = ProjectiveCamera::new(
            camera_base_params,
            lens_radius,
            focal_distance,
            screen_from_camera,
            screen_window
        );

        let dx_camera = projective.camera_from_raster.apply(Vec3f::new(1.0, 0.0, 0.0));
        let dy_camera = projective.camera_from_raster.apply(Vec3f::new(0.0, 1.0, 0.0));

        projective.camera_base.min_dir_differential_x = Vec3f::ZERO;
        projective.camera_base.min_dir_differential_y = Vec3f::ZERO;
        projective.camera_base.min_pos_differential_x = dx_camera;
        projective.camera_base.min_pos_differential_y = dy_camera;

        OrthographicCamera {
            projective,
            dx_camera,
            dy_camera,
        }
    }
}

impl AbstractCamera for OrthographicCamera {
    fn generate_ray(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective.camera_from_raster.apply(p_film);

        let mut ray = Ray::new_with_medium_time(
            p_camera,
            Vec3f::new(0.0, 0.0, 1.0),
            self.projective.camera_base.sample_time(sample.time),
            self.projective.camera_base.medium.clone(),
        );

        if self.projective.lens_radius > 0.0 {
            let p_lens = sample_uniform_disk_concentric(sample.p_lens) * self.projective.lens_radius;
            let ft = self.projective.focal_distance / ray.direction.z;
            let p_focus = ray.at(ft);

            ray.origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.direction = (p_focus - ray.origin).normalize().into();
        }

        Some(CameraRay::new(self.projective.camera_base.render_from_camera_ray(&ray)))
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective.camera_from_raster.apply(p_film);

        let mut ray = Ray::new_with_medium_time(
            p_camera,
            Vec3f::new(0.0, 0.0, 1.0),
            self.projective.camera_base.sample_time(sample.time),
            self.projective.camera_base.medium.clone(),
        );

        let p_lens = if self.projective.lens_radius > 0.0 {
            let p_lens = sample_uniform_disk_concentric(sample.p_lens) * self.projective.lens_radius;
            let ft = self.projective.focal_distance / ray.direction.z;
            let p_focus = ray.at(ft);

            ray.origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.direction = (p_focus - ray.origin).normalize().into();

            Some(p_lens)
        } else { None };

        let aux = if let Some(p_lens) = p_lens {
            let ft = self.projective.focal_distance / ray.direction.z;
            let p_focus = p_camera + self.dx_camera + (Vec3f::new(0.0, 0.0, 1.0) * ft);
            let rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let rx_direction = (p_focus - rx_origin).normalize().into();

            let p_focus = p_camera + self.dy_camera + (Vec3f::new(0.0, 0.0, 1.0) * ft);
            let ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let ry_direction = (p_focus - ry_origin).normalize().into();

            AuxiliaryRays::new(rx_origin, rx_direction, ry_origin, ry_direction)
        } else {
            AuxiliaryRays::new(ray.origin + self.dx_camera, ray.direction, ray.origin + self.dy_camera, ray.direction)
        };

        let rd = RayDifferential::new(ray, Some(aux));

        Some(CameraRayDifferential::new(rd))
    }

    fn get_film(&self) -> &Arc<Film> {
        &self.projective.camera_base.film
    }

    fn get_film_mut(&mut self) -> &mut Arc<Film> {
        &mut self.projective.camera_base.film
    }

    fn sample_time(&self, u: Float) -> Float {
        self.projective.camera_base.sample_time(u)
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        self.projective.init_metadata(metadata);
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        &self.projective.camera_base.camera_transform
    }

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vec3f, Vec3f) {
        self.projective.camera_base.approximate_dp_dxy(p, n, time, samples_per_pixel, options)
    }
}

#[derive(Debug, Clone)]
pub struct PerspectiveCamera {
    projective: ProjectiveCamera,
    dx_camera: Vec3f,
    dy_camera: Vec3f,
    cos_total_width: Float,
    area: Float,
}

impl PerspectiveCamera {
    pub fn create(
        parameters: &mut ParameterDictionary,
        camera_transform: CameraTransform,
        film: Arc<Film>,
        medium: Option<Arc<Medium>>,
        options: &Options,
        loc: &FileLoc
    ) -> PerspectiveCamera {
        let camera_base_params = CameraBaseParameters::new(camera_transform, film, medium, parameters, loc);

        let lens_radius = parameters.get_one_float("lensradius", 0.0);
        let focal_distance = parameters.get_one_float("focaldistance", 1e6);

        let x = camera_base_params.film.full_resolution().x as Float;
        let y = camera_base_params.film.full_resolution().y as Float;
        let frame = parameters.get_one_float("frameaspectratio", x / y);

        let mut screen = if frame > 1.0 {
            Bounds2f::new(Point2f::new(-frame, -1.0), Point2f::new(frame, 1.0))
        } else {
            Bounds2f::new(Point2f::new(-1.0, -1.0 / frame), Point2f::new(1.0, 1.0 / frame))
        };

        let sw = parameters.get_float_array("screenwindow");

        if !sw.is_empty() {
            if options.fullscreen {
                warn!(loc, "screenwindow is ignored in fullscreen mode");
            } else if sw.len() == 4 {
                screen = Bounds2f::new(Point2f::new(sw[0], sw[2]), Point2f::new(sw[1], sw[3]));
            } else {
                warn!(
                    loc,
                    "expected four values for 'screenwindow' parameter but got {}.",
                    sw.len(),
                );
            }
        }
        
        let fov = parameters.get_one_float("fov", 90.0);

        PerspectiveCamera::new(camera_base_params, fov, screen, lens_radius, focal_distance)
    }

    pub fn new(
        camera_base_params: CameraBaseParameters,
        fov: Float,
        screen_window: Bounds2f,
        lens_radius: Float,
        focal_distance: Float,
    ) -> PerspectiveCamera {
        let screen_from_camera = Transform::perspective(fov, 1e-2, 1000.0);
        let mut projective = ProjectiveCamera::new(
            camera_base_params,
            lens_radius,
            focal_distance,
            screen_from_camera,
            screen_window,
        );

        let dx_camera = Vec3f::from(projective.camera_from_raster.apply(Point3f::new(1.0, 0.0, 0.0))
            - projective.camera_from_raster.apply(Point3f::ZERO));
        let dy_camera = Vec3f::from(projective.camera_from_raster.apply(Point3f::new(0.0, 1.0, 0.0))
            - projective.camera_from_raster.apply(Point3f::ZERO));

        let radius = Point2f::from(projective.camera_base.film.get_filter().radius());
        let p_corner = Point3f::new(-radius.x, -radius.y, 0.0);
        let w_corner_camera = Vec3f::from((projective.camera_from_raster.apply(p_corner)).normalize());
        let cos_total_width = w_corner_camera.z;

        let res = projective.camera_base.film.full_resolution();
        let mut p_min = projective.camera_from_raster.apply(Point3f::ZERO);
        let mut p_max = projective.camera_from_raster.apply(Point3f::new(res.x as Float, res.y as Float, 0.0));

        p_min = p_min / p_min.z;
        p_max = p_max / p_max.z;

        let area = Float::abs((p_max.x - p_min.x) * (p_max.y - p_min.y));

        let camera = PerspectiveCamera {
            projective: projective.clone(),
            dx_camera,
            dy_camera,
            cos_total_width,
            area,
        };

        projective.camera_base.find_minimum_differentials(&camera);

        PerspectiveCamera {
            projective,
            dx_camera,
            dy_camera,
            cos_total_width,
            area,
        }
    }
}

impl AbstractCamera for PerspectiveCamera {
    fn generate_ray(&self, sample: &CameraSample, _lambda: &SampledWavelengths) -> Option<CameraRay> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective.camera_from_raster.apply(p_film);

        let mut ray = Ray::new_with_medium_time(
            Point3f::ZERO,
            Vec3f::from(p_camera).normalize(),
            self.sample_time(sample.time),
            self.projective.camera_base.medium.clone(),
        );

        if self.projective.lens_radius > 0.0 {
            let p_lens = sample_uniform_disk_concentric(sample.p_lens) * self.projective.lens_radius;
            let ft = self.projective.focal_distance / ray.direction.z;
            let p_focus = ray.at(ft);

            ray.origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.direction = (p_focus - ray.origin).normalize().into();
        }

        Some(CameraRay {
            ray: self.projective.camera_base.render_from_camera_ray(&ray),
            weight: SampledSpectrum::from_const(1.0),
        })
    }

    fn generate_ray_differential(&self, sample: &CameraSample, _lambda: &SampledWavelengths) -> Option<CameraRayDifferential> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective.camera_from_raster.apply(p_film);

        let dir = Vec3f::from(p_camera).normalize();
        let mut ray = Ray::new_with_medium_time(
            Point3f::ZERO,
            dir,
            self.sample_time(sample.time),
            self.projective.camera_base.medium.clone(),
        );

        let aux = if self.projective.lens_radius > 0.0 {
            let p_lens = self.projective.lens_radius * sample_uniform_disk_concentric(sample.p_lens);
            let ft = self.projective.focal_distance / ray.direction.z;
            let p_focus = ray.at(ft);

            ray.origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.direction = (p_focus - ray.origin).normalize().into();

            let dx = Vec3f::from(p_camera + self.dx_camera).normalize();
            let ft = self.projective.focal_distance / dx.z;
            let p_focus = Point3f::ZERO + ft * dx;
            let rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let rx_direction = (p_focus - rx_origin).normalize();

            let dy = Vec3f::from(p_camera + self.dy_camera).normalize();
            let ft = self.projective.focal_distance / dy.z;
            let p_focus = Point3f::ZERO + ft * dy;
            let ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let ry_direction = (p_focus - ry_origin).normalize();

            AuxiliaryRays::new(rx_origin, rx_direction.into(), ry_origin, ry_direction.into())
        } else {
            AuxiliaryRays::new(
                ray.origin,
                (Vec3f::from(p_camera) + self.dx_camera).normalize(),
                ray.origin,
                (Vec3f::from(p_camera) + self.dy_camera).normalize(),
            )
        };

        Some(CameraRayDifferential::new(
            self.projective.camera_base.render_from_camera_ray(&RayDifferential { ray, aux: Some(aux) })
        ))
    }

    fn get_film(&self) -> &Arc<Film> {
        self.projective.camera_base.get_film()
    }

    fn get_film_mut(&mut self) -> &mut Arc<Film> {
        self.projective.camera_base.get_film_mut()
    }

    fn sample_time(&self, u: Float) -> Float {
        self.projective.camera_base.sample_time(u)
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        self.projective.init_metadata(metadata);
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        &self.projective.camera_base.camera_transform
    }

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vec3f, Vec3f) {
        self.projective.camera_base.approximate_dp_dxy(p, n, time, samples_per_pixel, options)
    }
}
