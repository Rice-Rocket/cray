#![allow(clippy::excessive_precision)]
#![allow(clippy::too_many_arguments)]
#![allow(unused)]
#![feature(get_mut_unchecked)]

use crate::math::*;

pub mod math;
pub mod color;
pub mod file;
pub mod camera;
pub mod options;
pub mod media;
pub mod image;
pub mod reader;
pub mod texture;
pub mod mipmap;
pub mod shape;
pub mod primitive;
pub mod material;
pub mod light;
pub mod sampler;
pub mod bxdf;
pub mod bsdf;
pub mod phase;
pub mod integrator;
pub mod render;