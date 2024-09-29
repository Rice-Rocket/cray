use std::{cell::RefCell, sync::Arc};

use bumpalo::Bump;
use path::PathIntegrator;
use rand::{rngs::SmallRng, SeedableRng};
use random_walk::RandomWalkIntegrator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use simple_path::SimplePathIntegrator;
use simple_vol_path::SimpleVolumetricPathIntegrator;
use thread_local::ThreadLocal;
use vol_path::VolumetricPathIntegrator;

use crate::{camera::{film::{AbstractFilm, Film, VisibleSurface}, filter::get_camera_sample, AbstractCamera, Camera}, color::{colorspace::RgbColorSpace, rgb_xyz::Rgb, sampled::SampledSpectrum, wavelengths::SampledWavelengths}, error, image::ImageMetadata, interaction::Interaction, light::{sampler::{uniform::UniformLightSampler, LightSampler}, AbstractLight, Light, LightType}, material::{Material, MaterialEvalContext, SingleMaterial}, numeric::HasNan, options::Options, primitive::{AbstractPrimitive, Primitive}, reader::{error::ParseResult, paramdict::ParameterDictionary, target::FileLoc}, sampler::{AbstractSampler, Sampler}, shape::ShapeIntersection, tile::Tile, Float, Normal3f, Point2f, Point2i, Ray, RayDifferential, Vec3f};

pub mod random_walk;
pub mod simple_path;
pub mod path;
pub mod simple_vol_path;
pub mod vol_path;

pub trait AbstractIntegrator {
    fn render(&mut self, options: &Options);
}

pub enum Integrator {
    ImageTile(ImageTileIntegrator),
    Debug(DebugIntegrator),
}

impl Integrator {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
        color_space: Arc<RgbColorSpace>,
    ) -> ParseResult<Integrator> {
        Ok(match name {
            "randomwalk" => Integrator::ImageTile(ImageTileIntegrator::create_random_walk_integrator(
                parameters, camera, sampler, aggregate, lights,
            )?),
            "simplepath" => Integrator::ImageTile(ImageTileIntegrator::create_simple_path_integrator(
                parameters, camera, sampler, aggregate, lights,
            )?),
            "path" => Integrator::ImageTile(ImageTileIntegrator::create_path_integrator(
                parameters, camera, sampler, aggregate, lights,
            )?),
            "simplevolpath" => Integrator::ImageTile(ImageTileIntegrator::create_simple_vol_path_integrator(
                parameters, camera, sampler, aggregate, lights,
            )?),
            "volpath" => Integrator::ImageTile(ImageTileIntegrator::create_vol_path_integrator(
                parameters, camera, sampler, aggregate, lights,
            )?),
            "debug" => Integrator::Debug(DebugIntegrator::create(
                parameters, camera, sampler, aggregate, lights,
            )?),
            _ => { error!(@noloc "unknown integrator '{}'", name); },
        })
    }
}

impl AbstractIntegrator for Integrator {
    fn render(&mut self, options: &Options) {
        match self {
            Integrator::ImageTile(i) => i.render(options),
            Integrator::Debug(i) => i.render(options),
        }
    }
}

pub struct FilmSample {
    p_film: Point2i,
    l: SampledSpectrum,
    lambda: SampledWavelengths,
    visible_surface: Option<VisibleSurface>,
    weight: Float,
}

pub struct IntegratorBase {
    aggregate: Arc<Primitive>,
    lights: Arc<[Arc<Light>]>,
    infinite_lights: Vec<Arc<Light>>,
}

impl IntegratorBase {
    const SHADOW_EPSILON: Float = 0.0001;
    
    pub fn new(aggregate: Arc<Primitive>, mut lights: Arc<[Arc<Light>]>) -> IntegratorBase {
        let scene_bounds = aggregate.bounds();

        // TODO: Instead of using nightly feature (get_mut_unchecked), use interior mutability
        unsafe {
            for light in Arc::get_mut_unchecked(&mut lights).iter_mut() {
                Arc::get_mut_unchecked(light).preprocess(&scene_bounds);
            }
        }

        let mut infinite_lights = Vec::new();

        for light in lights.as_ref() {
            if light.light_type() == LightType::Infinite {
                infinite_lights.push(light.clone());
            }
        }

        IntegratorBase {
            aggregate,
            lights,
            infinite_lights,
        }
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        debug_assert!(ray.direction != Vec3f::ZERO);
        self.aggregate.intersect(ray, t_max)
    }

    pub fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        debug_assert!(ray.direction != Vec3f::ZERO);
        self.aggregate.intersect_predicate(ray, t_max)
    }

    pub fn unoccluded(&self, p0: &Interaction, p1: &Interaction) -> bool {
        !self.intersect_predicate(&p0.spawn_ray_to_interaction(p1), 1.0 - Self::SHADOW_EPSILON)
    }
}

pub struct ImageTileIntegrator {
    base: IntegratorBase,
    camera: Camera,
    sampler_prototype: Sampler,

    ray_integrator: RayIntegrator,
}

impl ImageTileIntegrator {
    pub fn new(
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
        camera: Camera,
        sampler: Sampler,
        ray_integrator: RayIntegrator,
    ) -> ImageTileIntegrator {
        ImageTileIntegrator {
            base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            ray_integrator,
        }
    }

    pub fn create_random_walk_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<ImageTileIntegrator> {
        let max_depth = parameters.get_one_int("maxdepth", 5)?;
        let ray_integrator = RayIntegrator::RandomWalk(RandomWalkIntegrator { max_depth });

        Ok(ImageTileIntegrator::new(
            aggregate,
            lights,
            camera,
            sampler,
            ray_integrator,
        ))
    }

    pub fn create_simple_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<ImageTileIntegrator> {
        let max_depth = parameters.get_one_int("maxdepth", 5)?;
        let sample_lights = parameters.get_one_bool("samplelights", true)?;
        let sample_bsdf = parameters.get_one_bool("samplebsdf", true)?;
        let light_sampler = UniformLightSampler { lights: lights.clone() };

        let pixel_sample_evaluator = RayIntegrator::SimplePath(SimplePathIntegrator {
            max_depth,
            sample_lights,
            sample_bsdf,
            light_sampler,
        });

        Ok(ImageTileIntegrator::new(
            aggregate,
            lights,
            camera,
            sampler,
            pixel_sample_evaluator,
        ))
    }

    pub fn create_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<ImageTileIntegrator> {
        let max_depth = parameters.get_one_int("maxdepth", 5)?;
        let regularize = parameters.get_one_bool("regularize", false)?;

        let light_strategy = parameters.get_one_string("lightsampler", "bvh")?;
        let light_sampler = LightSampler::create(&light_strategy, lights.clone())?;

        let pixel_sample_evaluator = RayIntegrator::Path(PathIntegrator::new(
            max_depth,
            light_sampler,
            regularize,
        )?);

        Ok(ImageTileIntegrator::new(
            aggregate,
            lights,
            camera,
            sampler,
            pixel_sample_evaluator,
        ))
    }

    pub fn create_simple_vol_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<ImageTileIntegrator> {
        let max_depth = parameters.get_one_int("maxdepth", 5)?;
        let ray_integrator = RayIntegrator::SimpleVolumetricPath(SimpleVolumetricPathIntegrator { max_depth });

        Ok(ImageTileIntegrator::new(
            aggregate,
            lights,
            camera,
            sampler,
            ray_integrator,
        ))
    }

    pub fn create_vol_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<ImageTileIntegrator> {
        let max_depth = parameters.get_one_int("maxdepth", 5)?;
        let regularize = parameters.get_one_bool("regularize", false)?;

        let light_strategy = parameters.get_one_string("lightsampler", "bvh")?;
        let light_sampler = LightSampler::create(&light_strategy, lights.clone())?;

        let ray_integrator = RayIntegrator::VolumetricPath(VolumetricPathIntegrator::new(
            max_depth,
            light_sampler,
            regularize,
        )?);

        Ok(ImageTileIntegrator::new(
            aggregate,
            lights,
            camera,
            sampler,
            ray_integrator,
        ))
    }
}

impl AbstractIntegrator for ImageTileIntegrator {
    fn render(&mut self, options: &Options) {
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        let tiles = Tile::tiles(pixel_bounds, 8, 8);

        let scratch_buffer_tl = ThreadLocal::new();
        let sampler_tl = ThreadLocal::new();
        let mut film = self.camera.get_film_mut().clone();

        let mut n_waves = 0;
        while wave_start < spp {
            wave_start = wave_end;
            wave_end = i32::min(spp, wave_end + next_wave_size);
            next_wave_size = i32::min(2 * next_wave_size, 64);
            n_waves += 1;
        };

        wave_start = 0;
        wave_end = 1;
        next_wave_size = 1;

        let mut wave_index = 0;

        let bar_template = "{spinner:.green} [{elapsed}] [{bar:50.white/white}] {pos}/{len} ({eta})";
        let style = ProgressStyle::with_template(bar_template).unwrap()
            .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");

        while wave_start < spp {
            wave_index += 1;

            tiles.par_iter()
                .progress_with_style(style.clone().template(&format!(
                    "{} {}",
                    console::style(format!("Wave {}/{}", wave_index, n_waves)).green().bold(),
                    bar_template,
                )).unwrap())
                .for_each(|tile| {
                    let scratch_buffer = scratch_buffer_tl.get_or(|| RefCell::new(Bump::with_capacity(256)));
                    let sampler = sampler_tl.get_or(|| RefCell::new(self.sampler_prototype.clone()));

                    let mut rng = SmallRng::from_entropy();

                    for x in tile.bounds.min.x..tile.bounds.max.x {
                        for y in tile.bounds.min.y..tile.bounds.max.y {
                            let p_pixel = Point2i::new(x, y);
                            for sample_index in wave_start..wave_end {
                                sampler.borrow_mut().start_pixel_sample(p_pixel, sample_index, 0);
                                let film_sample = self.evaluate_pixel_sample(
                                    &self.base,
                                    &self.camera,
                                    p_pixel,
                                    sample_index,
                                    &mut sampler.borrow_mut(),
                                    &mut scratch_buffer.borrow_mut(),
                                    options,
                                    &mut rng,
                                );

                                unsafe {
                                    Arc::get_mut_unchecked(&mut film.clone()).add_sample(
                                        film_sample.p_film,
                                        &film_sample.l,
                                        &film_sample.lambda,
                                        &film_sample.visible_surface,
                                        film_sample.weight,
                                    );
                                }

                                scratch_buffer.borrow_mut().reset();
                            }
                        }
                    }
                });

            wave_start = wave_end;
            wave_end = i32::min(spp, wave_end + next_wave_size);
            next_wave_size = i32::min(2 * next_wave_size, 64);

            if wave_start == spp {
                let mut metadata = ImageMetadata::default();
                self.camera.get_film().write_image(&mut metadata, 1.0 / wave_start as Float).unwrap();
            }
        }
    }
}

impl ImageTileIntegrator {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> FilmSample {
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            sampler.get_1d()
        };

        let mut lambda = camera.get_film().sample_wavelengths(lu);
        let res = camera.get_film().full_resolution();

        let camera_sample = get_camera_sample(sampler, p_pixel, camera.get_film().get_filter(), options);
        let camera_ray = camera.generate_ray_differential(&camera_sample, &lambda);

        let l = if let Some(mut camera_ray) = camera_ray {
            debug_assert!(camera_ray.ray.ray.direction.length() > 0.999);
            debug_assert!(camera_ray.ray.ray.direction.length() < 1.001);

            let ray_diff_scale = Float::max(
                0.125,
                1.0 / Float::sqrt(sampler.samples_per_pixel() as Float),
            );

            if !options.disable_pixel_jitter {
                camera_ray.ray.scale_differentials(ray_diff_scale);
            }

            let mut l = camera_ray.weight * self.ray_integrator.li(
                base,
                camera,
                &mut camera_ray.ray,
                &mut lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            );

            if l.has_nan() {
                l = SampledSpectrum::from_const(0.0);
                // error!(@panic "ray integrator produced NaN value");
            } else if l.y(&lambda).is_infinite() {
                l = SampledSpectrum::from_const(0.0);
                // error!(@panic "ray integrator produced infinite value");
            }

            l
        } else {
            SampledSpectrum::from_const(0.0)
        };

        FilmSample {
            p_film: p_pixel,
            l,
            lambda,
            visible_surface: None,
            weight: camera_sample.filter_weight,
        }
    }
}

pub enum DebugMode {
    Depth,
    Normal,
    UV,
    Material,
}

pub struct DebugIntegrator {
    base: IntegratorBase,
    camera: Camera,
    sampler_prototype: Sampler,
    mode: DebugMode,
}

impl DebugIntegrator {
    pub fn create(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
    ) -> ParseResult<DebugIntegrator> {
        let mode = parameters.get_one_string("mode", "normal")?;
        
        let mode = match mode.as_str() {
            "depth" => DebugMode::Depth,
            "normal" => DebugMode::Normal,
            "uv" => DebugMode::UV,
            "material" => DebugMode::Material,
            s => { error!(@noloc "unknown debug mode '{}'", s); },
        };
        
        let Film::Debug(_) = camera.get_film().as_ref() else {
            error!(@noloc "debug integrator must be used with debug film");
        };

        Ok(DebugIntegrator {
            base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            mode,
        })
    }
}

impl AbstractIntegrator for DebugIntegrator {
    fn render(&mut self, options: &Options) {
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();
        let mut film = self.camera.get_film_mut().clone();

        let sampler = RefCell::new(self.sampler_prototype.clone());
        for x in pixel_bounds.min.x..pixel_bounds.max.x {
            for y in pixel_bounds.min.y..pixel_bounds.max.y {
                let p_pixel = Point2i::new(x, y);

                let rgb = self.evaluate_pixel_sample(
                    &self.base,
                    &self.camera,
                    p_pixel,
                    &mut sampler.borrow_mut(),
                    options,
                );

                unsafe {
                    if let Film::Debug(f) = Arc::get_mut_unchecked(&mut film.clone()) {
                        f.add_pixel(p_pixel, rgb);
                    }
                }
            }
        }

        let mut metadata = ImageMetadata::default();
        self.camera.get_film().write_image(&mut metadata, 1.0).unwrap();
    }
}

impl DebugIntegrator {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sampler: &mut Sampler,
        options: &Options,
    ) -> Rgb {
        let lambda = camera.get_film().sample_wavelengths(0.5);
        let res = camera.get_film().full_resolution();
        
        let camera_sample = get_camera_sample(sampler, p_pixel, camera.get_film().get_filter(), options);
        let camera_ray = camera.generate_ray_differential(&camera_sample, &lambda);

        let isect = camera_ray.and_then(|camera_ray| base.intersect(&camera_ray.ray.ray, Float::INFINITY));

        match self.mode {
            DebugMode::Depth => {
                if let Some(i) = isect {
                    let dist = 1.0 / i.t_hit;
                    Rgb::new(dist, dist, dist)
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            },
            DebugMode::Normal => {
                if let Some(i) = isect {
                    let normal = i.intr.interaction.n;
                    Rgb::new(normal.x, normal.y, normal.z) * 0.5 + 0.5
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            },
            DebugMode::UV => {
                if let Some(i) = isect {
                    let uv = i.intr.interaction.uv;
                    Rgb::new(uv.x, uv.y, 0.0)
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            },
            DebugMode::Material => {
                if let Some(mat) = isect.as_ref().and_then(|i| i.intr.material.clone()) {
                    let Material::Single(SingleMaterial::Debug(m)) = mat.as_ref() else {
                        panic!("can only use debug material with debug integrator in material mode");
                    };

                    let ctx = MaterialEvalContext::from(&isect.unwrap().intr);
                    m.get_color(&ctx)
                } else {
                    Rgb::new(0.0, 0.0, 0.0)
                }
            }
        }
    }
}

pub trait AbstractRayIntegrator {
    fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum;
}

pub enum RayIntegrator {
    RandomWalk(RandomWalkIntegrator),
    SimplePath(SimplePathIntegrator),
    Path(PathIntegrator),
    SimpleVolumetricPath(SimpleVolumetricPathIntegrator),
    VolumetricPath(VolumetricPathIntegrator),
}

impl AbstractRayIntegrator for RayIntegrator {
    fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum {
        match self {
            RayIntegrator::RandomWalk(r) => r.li(
                base,
                camera,
                ray,
                lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            ),
            RayIntegrator::SimplePath(r) => r.li(
                base,
                camera,
                ray,
                lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            ),
            RayIntegrator::Path(r) => r.li(
                base,
                camera,
                ray,
                lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            ),
            RayIntegrator::SimpleVolumetricPath(r) => r.li(
                base,
                camera,
                ray,
                lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            ),
            RayIntegrator::VolumetricPath(r) => r.li(
                base,
                camera,
                ray,
                lambda,
                sampler,
                scratch_buffer,
                options,
                rng,
            ),
        }
    }
}
