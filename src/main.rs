use std::{collections::HashMap, path::PathBuf, sync::{Arc, Mutex}, time::Instant};

use clap::Parser;
use console::style;
use cray::{
    color::rgb_xyz::ColorEncodingCache, math::{Bounds2f, Bounds2i, Float, Point2f, Point2i}, options::{CameraRenderingSpace, Options}, reader::{parser::parse_files, scene::{BasicScene, BasicSceneBuilder}}, render::render_cpu
};
use string_interner::StringInterner;

#[derive(clap::Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Render the given scene.
    scene_file: String,

    /// Write the final image to the given filename.
    #[arg(short, long)]
    outfile: Option<String>,

    /// Specify an image crop window [0,1]^2. <x0 x1 y0 y1>
    #[arg(long, num_args = 4)]
    crop_window: Option<Vec<Float>>,

    /// Specify an image crop window from pixel coordinates. <x0 x1 y0 y1>
    #[arg(long, num_args = 4)]
    pixel_bounds: Option<Vec<i32>>,

    /// Force all materials to be diffuse.
    #[arg(long, default_value = "false")]
    force_diffuse: bool,

    /// Set the seed of the random number generator.
    #[arg(long, default_value = "0")]
    seed: i32,

    /// Override the samples per pixel given by the scene file.
    #[arg(short, long)]
    spp: Option<i32>,

    /// Automatically disable some quality settings to render more quickly.
    #[arg(short, long)]
    quick: bool,

    /// Always sample the same wavelengths of light.
    #[arg(long)]
    disable_wavelength_jitter: bool,

    /// Always sample pixels at their centers.
    #[arg(long)]
    disable_pixel_jitter: bool,

    /// Point-sample all textures.
    #[arg(long)]
    disable_texture_filtering: bool,

    /// Set the rendering coordinate system.
    #[arg(long, default_value = "camera-world")]
    render_coord_system: CameraRenderingSpace,

    /// Fullscreen
    #[arg(long, default_value = "false")]
    fullscreen: bool,

    /// Scale the target triangle edge length by the given value.
    #[arg(long, default_value = "1.0")]
    displacement_edge_scale: Float,

    /// Reference image for MSE calculations.
    #[arg(long)]
    mse_reference_image: Option<String>,

    /// Output MSE statistics to given filename.
    #[arg(long)]
    mse_reference_output: Option<String>,

    /// Record pixel statistics
    #[arg(long, default_value = "false")]
    record_pixel_statistics: bool,

    /// Use wavefront volumentric path integrator.
    #[arg(short, long, default_value = "false")]
    wavefront: bool,

    /// Directory to search for files.
    #[arg(long)]
    search_directory: Option<String>,
}

fn main() {
    let cli = Args::parse();

    let mut options = Options::default();
    if let Some(crop_window) = cli.crop_window {
        options.crop_window = Some(Bounds2f::new(
            Point2f::new(crop_window[0], crop_window[1]),
            Point2f::new(crop_window[2], crop_window[3]),
        ));
    }

    options.force_diffuse = cli.force_diffuse;
    options.seed = cli.seed;
    if let Some(pixel_bounds) = cli.pixel_bounds {
        options.pixel_bounds = Some(Bounds2i::new(
            Point2i::new(pixel_bounds[0], pixel_bounds[1]),
            Point2i::new(pixel_bounds[2], pixel_bounds[3]),
        ));
    }

    options.image_file = cli.outfile;
    if let Some(spp) = cli.spp {
        options.pixel_samples = Some(spp);
    } else {
        options.pixel_samples = None;
    }

    options.quick_render = cli.quick;
    if options.quick_render {
        todo!("Quick render is not yet implemented");
    }

    options.disable_wavelength_jitter = cli.disable_wavelength_jitter;
    options.disable_pixel_jitter = cli.disable_pixel_jitter;
    options.disable_texture_filtering = cli.disable_texture_filtering;
    options.rendering_space = cli.render_coord_system;
    options.fullscreen = cli.fullscreen;

    if options.fullscreen {
        todo!("Fullscreen is not yet implemented");
    }

    options.displacement_edge_scale = cli.displacement_edge_scale;
    if cli.displacement_edge_scale != 1.0 {
        todo!("Displacement edge scale is not yet implemented");
    }
    options.mse_reference_image = cli.mse_reference_image;
    if options.mse_reference_image.is_some() {
        todo!("MSE reference image is not yet implemented.")
    }
    options.mse_reference_output = cli.mse_reference_output;
    if options.mse_reference_output.is_some() {
        todo!("MSE reference output is not yet implemented.")
    }
    options.record_pixel_statistics = cli.record_pixel_statistics;
    if options.record_pixel_statistics {
        todo!("Record pixel statistics is not yet implemented.")
    }
    options.wavefront = cli.wavefront;
    if options.wavefront {
        todo!("Wavefront is not yet implemented.")
    }

    if let Some(search_directory) = cli.search_directory {
        options.search_directory = Some(PathBuf::from(search_directory));
    } else {
        options.search_directory = None;
    }
    if options.search_directory.is_some() {
        todo!("Search directory is not yet implemented.")
    }

    let mut string_interner = StringInterner::new();
    let mut cached_spectra = HashMap::new();
    let texture_cache = Arc::new(Mutex::new(HashMap::new()));
    let mut gamma_encoding_cache = ColorEncodingCache::new();

    let scene = Box::new(BasicScene::default());
    let mut scene_builder = BasicSceneBuilder::new(scene, &mut string_interner, &options);
    
    println!(
        "{} Reading scene '{}/{}'...",
        style("[1/7]").bold().dim(),
        PathBuf::from(&cli.scene_file).components().nth_back(1).and_then(|c| c.as_os_str().to_str()).unwrap_or(""),
        PathBuf::from(&cli.scene_file).file_name().and_then(|s| s.to_str()).unwrap_or(""),
    );

    let parse_result = parse_files(
        &[&cli.scene_file],
        &mut scene_builder,
        &mut options,
        &mut string_interner,
        &mut cached_spectra,
        &texture_cache,
        &mut gamma_encoding_cache,
    );

    match parse_result {
        Ok(_) => (),
        Err(err) => {
            if let Some(msg) = err.msg {
                println!("error: {}", msg);
            }
            return;
        }
    }

    let scene = scene_builder.done();
    let start_time = Instant::now();
    let render_result = render_cpu(scene, &options, &mut string_interner, &mut cached_spectra, &texture_cache, &mut gamma_encoding_cache);
    let elapsed = start_time.elapsed();

    match render_result {
        Ok(_) => (),
        Err(err) => {
            if let Some(msg) = err.msg {
                println!("error: {}", msg);
            }
            return;
        }
    }

    println!(
        "Finished rendering in {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
    );
}
