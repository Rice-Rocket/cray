use std::path::PathBuf;

use crate::{Bounds2f, Bounds2i};

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
}
