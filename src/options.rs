use std::path::PathBuf;

use crate::{Bounds2f, Bounds2i, Float};

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum CameraRenderingSpace {
    Camera,
    CameraWorld,
    World,
}

pub struct Options {
    pub seed: i32,
    pub disable_pixel_jitter: bool,
    pub disable_wavelength_jitter: bool,
    pub disable_texture_filtering: bool,
    pub force_diffuse: bool,
    pub wavefront: bool,
    pub rendering_space: CameraRenderingSpace,
    pub pixel_samples: Option<i32>,

    pub disable_image_textures: bool,

    pub search_directory: Option<PathBuf>,
    pub image_file: Option<String>,

    pub pixel_bounds: Option<Bounds2i>,
    pub crop_window: Option<Bounds2f>,
    pub quick_render: bool,
    pub fullscreen: bool,

    pub mse_reference_image: Option<String>,
    pub mse_reference_output: Option<String>,
    pub record_pixel_statistics: bool,
    pub displacement_edge_scale: Float,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0,
            rendering_space: CameraRenderingSpace::World,
            disable_texture_filtering: false,
            disable_pixel_jitter: false,
            disable_wavelength_jitter: false,
            disable_image_textures: false,
            force_diffuse: false,
            image_file: None,
            quick_render: false,
            pixel_bounds: None,
            crop_window: None,
            pixel_samples: None,
            fullscreen: false,
            displacement_edge_scale: 1.0,
            mse_reference_image: None,
            mse_reference_output: None,
            record_pixel_statistics: false,
            wavefront: false,
            search_directory: None,
        }
    }
}
