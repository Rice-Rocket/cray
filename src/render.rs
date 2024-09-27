use std::{collections::HashMap, sync::{Arc, Mutex}};

use console::style;
use string_interner::{DefaultBackend, StringInterner};

use crate::{color::{rgb_xyz::ColorEncodingCache, spectrum::Spectrum}, integrator::AbstractIntegrator, mipmap::MIPMap, options::Options, reader::scene::BasicScene, texture::TexInfo};

pub fn render_cpu(
    mut scene: Box<BasicScene>,
    options: &Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
) {
    println!(
        "{} Creating media...",
        style("[2/7]").bold().dim(),
    );
    let media = scene.create_media();

    println!(
        "{} Creating textures...",
        style("[3/7]").bold().dim(),
    );
    let textures = scene.create_textures(cached_spectra, string_interner, options, texture_cache, gamma_encoding_cache);

    println!(
        "{} Creating lights...",
        style("[4/7]").bold().dim(),
    );
    let (lights, shape_index_to_area_lights) = scene.create_lights(&textures, string_interner, options);

    println!(
        "{} Creating materials...",
        style("[5/7]").bold().dim(),
    );
    let (named_materials, materials) = scene.create_materials(&textures, string_interner, cached_spectra, options);

    println!(
        "{} Creating accelerator...",
        style("[6/7]").bold().dim(),
    );
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

    println!(
        "{} Rendering image...",
        style("[7/7]").bold().dim(),
    );
    // TODO: check options, give warnings.
    let mut integrator = scene.create_integrator(camera, sampler, accelerator, lights, string_interner);
    integrator.render(options);
}
