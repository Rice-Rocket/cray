use std::{collections::HashMap, sync::{Arc, Mutex}};

use string_interner::{DefaultBackend, StringInterner};
use tracing::info;

use crate::{color::{rgb_xyz::ColorEncodingCache, spectrum::Spectrum}, integrator::AbstractIntegrator, mipmap::MIPMap, options::Options, reader::scene::BasicScene, texture::TexInfo};

pub fn render_cpu(
    mut scene: Box<BasicScene>,
    options: &Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
) {
    info!("Creating media...");
    let media = scene.create_media();
    info!("Done creating media.");

    info!("Creating textures...");
    let textures = scene.create_textures(cached_spectra, string_interner, options, texture_cache, gamma_encoding_cache);
    info!("Done creating textures.");

    info!("Creating lights...");
    let (lights, shape_index_to_area_lights) = scene.create_lights(&textures, string_interner, options);
    info!("Done creating lights.");

    info!("Creating materials...");
    let (named_materials, materials) = scene.create_materials(&textures, string_interner, cached_spectra, options);
    info!("Done creating materials.");

    let accelerator = scene.create_aggregate(
        &textures,
        &shape_index_to_area_lights,
        &media,
        &named_materials,
        &materials,
        string_interner,
        options,
    );

    let camera = scene.get_camera().unwrap();
    let sampler = scene.get_sampler().unwrap();

    // TODO: check options, give warnings.
    let mut integrator = scene.create_integrator(camera, sampler, accelerator, lights, string_interner);
    integrator.render(options);
}
