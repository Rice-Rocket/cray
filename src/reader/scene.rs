use std::{collections::{HashMap, HashSet}, ops::{Index, IndexMut}, path::{Path, PathBuf}, str::FromStr as _, sync::{atomic::{AtomicI32, Ordering}, Arc, Mutex}};

use string_interner::{symbol::SymbolU32, DefaultBackend, StringInterner};

use crate::{camera::{film::{AbstractFilm, Film}, filter::Filter, AbstractCamera, Camera, CameraTransform}, clear_log, color::{colorspace::{NamedColorSpace, RgbColorSpace}, rgb_xyz::{ColorEncoding, ColorEncodingCache}, spectrum::Spectrum}, error, file::resolve_filename, image::Image, integrator::{AbstractIntegrator, Integrator}, light::Light, log, material::Material, media::{Medium, MediumInterface}, mipmap::MIPMap, options::{CameraRenderingSpace, Options}, primitive::{bvh::{create_accelerator, BvhAggregate, BvhSplitMethod}, geometric::GeometricPrimitive, simple::SimplePrimitive, transformed::TransformedPrimitive, Primitive}, sampler::Sampler, shape::Shape, texture::{FloatConstantTexture, FloatTexture, SpectrumTexture, TexInfo}, to_radians, transform::{ApplyTransform, Transform}, warn, Float, Mat4, Point3f, Vec3f};

use super::{error::ParseResult, paramdict::{NamedTextures, ParameterDictionary, SpectrumType, TextureParameterDictionary}, target::{FileLoc, ParsedParameterVector, ParserTarget}, utils::normalize_arg};

#[derive(Debug, Default)]
pub struct BasicScene {
    pub integrator: Option<SceneEntity>,
    pub accelerator: Option<SceneEntity>,
    pub film_color_space: Option<Arc<RgbColorSpace>>,
    pub shapes: Vec<ShapeSceneEntity>,
    pub instances: Vec<InstanceSceneEntity>,
    pub instance_definitions: Vec<InstanceDefinitionSceneEntity>,
    camera: Option<Camera>,
    film: Option<Film>,
    sampler: Option<Sampler>,
    named_materials: Vec<(String, SceneEntity)>,
    materials: Vec<SceneEntity>,
    area_lights: Vec<SceneEntity>,
    normal_maps: HashMap<String, Arc<Image>>,
    media: HashMap<String, Arc<Medium>>,
    serial_float_textures: Vec<(String, TextureSceneEntity)>,
    serial_spectrum_textures: Vec<(String, TextureSceneEntity)>,
    async_spectrum_textures: Vec<(String, TextureSceneEntity)>,
    loading_texture_filenames: HashSet<String>,

    textures: NamedTextures,
    lights: Vec<Arc<Light>>,
}

impl BasicScene {
    pub fn get_camera(&self) -> Option<Camera> {
        self.camera.clone()
    }

    pub fn get_film(&self) -> Option<Film> {
        self.film.clone()
    }

    pub fn get_sampler(&self) -> Option<Sampler> {
        self.sampler.clone()
    }

    pub fn set_options(
        &mut self,
        mut filter: SceneEntity,
        mut film: SceneEntity,
        mut camera: CameraSceneEntity,
        mut sampler: SceneEntity,
        integ: SceneEntity,
        accel: SceneEntity,
        string_interner: &StringInterner<DefaultBackend>,
        options: &Options,
    ) -> ParseResult<()> {
        self.film_color_space = Some(film.parameters.color_space.clone());
        self.integrator = Some(integ);
        self.accelerator = Some(accel);

        let filter = Filter::create(
            string_interner.resolve(filter.name).expect("unresolved name"),
            &mut filter.parameters,
            &filter.loc,
        )?;

        let exposure_time = camera.base.parameters.get_one_float("shutterclose", 1.0)?
            - camera.base.parameters.get_one_float("shutteropen", 0.0)?;

        if exposure_time <= 0.0 {
            error!(
                camera.base.loc,
                "the specified camera shutter times imply the camera won't open",
            );
        }

        self.film = Some(Film::create(
            string_interner.resolve(film.name).unwrap(),
            &mut film.parameters,
            exposure_time,
            &camera.camera_transform,
            filter,
            &film.loc,
            options,
        )?);

        let res = self.film.as_ref().unwrap().full_resolution();
        self.sampler = Some(Sampler::create(
            string_interner.resolve(sampler.name).unwrap(),
            &mut sampler.parameters,
            res,
            options,
            &sampler.loc,
        )?);

        self.camera = Some(Camera::create(
            string_interner.resolve(camera.base.name).unwrap(),
            &mut camera.base.parameters,
            self.get_medium(&camera.medium),
            camera.camera_transform,
            Arc::new(self.film.as_ref().unwrap().clone()),
            options,
            &camera.base.loc,
        )?);

        Ok(())
    }

    fn add_named_material(&mut self, name: &str, mut material: SceneEntity, options: &Options) {
        self.load_normal_map(&mut material.parameters, options);
        self.named_materials.push((name.to_owned(), material));
    }

    fn add_material(&mut self, mut material: SceneEntity, options: &Options) -> i32 {
        self.load_normal_map(&mut material.parameters, options);
        self.materials.push(material);
        (self.materials.len() - 1) as i32
    }

    fn get_medium(&mut self, name: &str) -> Option<Arc<Medium>> {
        self.media.get(name).cloned()
    }

    fn add_medium(
        &mut self,
        mut medium: MediumSceneEntity,
        string_interner: &mut StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<()> {
        let ty = medium.base.parameters.get_one_string("type", "")?;
        if ty.is_empty() {
            error!(medium.base.loc, "no parameter \"string type\" found for medium");
        }

        let m = Arc::new(Medium::create(
            &ty,
            &mut medium.base.parameters,
            medium.render_from_object,
            cached_spectra,
            &medium.base.loc,
        )?);

        self.media.insert(string_interner.resolve(medium.base.name).expect("Unknown symbol").to_owned(), m);

        Ok(())
    }

    fn add_float_texture(
        &mut self,
        name: &str,
        mut texture: TextureSceneEntity,
        string_interner: &StringInterner<DefaultBackend>,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> ParseResult<()> {
        // TODO: Check if animated once we add animated transforms.

        if string_interner
            .resolve(texture.base.name)
            .expect("Unknown texture name")
            != "imagemap"
            && string_interner
                .resolve(texture.base.name)
                .expect("Unknown texture name")
                != "ptex"
        {
            self.serial_float_textures.push((name.to_owned(), texture));
            return Ok(());
        }

        let filename = texture
            .base
            .parameters
            .get_one_string("filename", "")?;

        let filename = resolve_filename(options, filename.as_str());
        if filename.is_empty() {
            error!(texture.base.loc, "no filename provided for texture.");
        }

        let path = Path::new(filename.as_str());
        if !path.exists() {
            error!(texture.base.loc, "texture \"{}\" not found.", filename);
        }

        if self.loading_texture_filenames.contains(&filename) {
            self.serial_float_textures.push((name.to_owned(), texture));
            return Ok(());
        }

        self.loading_texture_filenames.insert(filename);

        // TODO: Can make this async.
        let render_from_texture = texture.render_from_object;

        let mut tex_dict = TextureParameterDictionary::new(texture.base.parameters.clone());
        let float_texture = FloatTexture::create(
            string_interner
                .resolve(texture.base.name)
                .expect("Unknown symbol"),
            render_from_texture,
            &mut tex_dict,
            &texture.base.loc,
            &self.textures,
            options,
            texture_cache,
            gamma_encoding_cache,
        )?;

        self.textures
            .float_textures
            .insert(name.to_owned(), Arc::new(float_texture));

        Ok(())
    }

    fn add_spectrum_texture(
        &mut self,
        name: &str,
        mut texture: TextureSceneEntity,
        string_interner: &StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> ParseResult<()> {
        if string_interner.resolve(texture.base.name).unwrap() != "ptex"
            && string_interner.resolve(texture.base.name).unwrap() != "imagemap"
        {
            self.serial_spectrum_textures
                .push((name.to_owned(), texture));
            return Ok(());
        }

        let filename = texture
            .base
            .parameters
            .get_one_string("filename", "")?;
        let filename = resolve_filename(options, filename.as_str());

        if filename.is_empty() {
            error!(texture.base.loc, "no filename provided for texture.");
        }

        let path = Path::new(&filename);
        if !path.exists() {
            error!(texture.base.loc, "texture \"{}\" not found.", filename);
        }

        if self.loading_texture_filenames.contains(&filename) {
            self.serial_spectrum_textures
                .push((name.to_owned(), texture));
            return Ok(());
        }
        self.loading_texture_filenames.insert(filename);

        self.async_spectrum_textures.push((name.to_owned(), texture.clone()));

        let render_from_texture = texture.render_from_object;
        let mut text_dict = TextureParameterDictionary::new(texture.base.parameters.clone());
        let spectrum_texture = SpectrumTexture::create(
            string_interner
                .resolve(texture.base.name)
                .expect("Unknown symbol"),
            render_from_texture,
            &mut text_dict,
            SpectrumType::Albedo,
            cached_spectra,
            &self.textures,
            &texture.base.loc,
            options,
            texture_cache,
            gamma_encoding_cache,
        )?;

        self.textures
            .albedo_spectrum_textures
            .insert(name.to_owned(), Arc::new(spectrum_texture));

        Ok(())
    }

    fn add_light(
        &mut self,
        light: &mut LightSceneEntity,
        string_interner: &StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) -> ParseResult<()> {
        let light_medium = self.get_medium(&light.medium);
        // TODO: Check for animated light and warn, when I add animated transforms.

        // TODO: Change to async, or otherwise parallelize (i.e. could place these
        // params into a vec, and consume that vec in a par_iter() in create_lights).
        self.lights.push(Arc::new(Light::create(
            string_interner
                .resolve(light.base.base.name)
                .expect("Unknown symbol"),
            &mut light.base.base.parameters,
            light.base.render_from_object,
            self.get_camera().unwrap().get_camera_transform(),
            None,
            &light.base.base.loc,
            cached_spectra,
            options,
        )?));

        Ok(())
    }

    fn add_area_light(&mut self, light: SceneEntity) -> i32 {
        self.area_lights.push(light);
        (self.area_lights.len() - 1) as i32
    }

    fn add_shapes(&mut self, shapes: &[ShapeSceneEntity]) {
        self.shapes.extend_from_slice(shapes);
    }

    // TODO: add_animated_shapes().

    fn add_instance_definition(&mut self, instance: InstanceDefinitionSceneEntity) {
        self.instance_definitions.push(instance);
    }

    fn add_instance_uses(&mut self, instances: &[InstanceSceneEntity]) {
        self.instances.extend_from_slice(instances);
    }

    fn done(&mut self) {
        // TODO: Check for unused textures, lights, etc and warn about them.
    }

    fn load_normal_map(&mut self, parameters: &mut ParameterDictionary, options: &Options) -> ParseResult<()> {
        let normal_map_filename = resolve_filename(options, &parameters.get_one_string("normalmap", "")?);
        if normal_map_filename.is_empty() {
            return Ok(());
        }
        let filename = PathBuf::from(&normal_map_filename);
        if !filename.exists() {
            warn!(@image filename.display(), "normal map not found.");
        }

        let image_and_metadata = Image::read(
            &filename,
            Some(ColorEncoding::get("linear", None)?),
        )?;

        let image = image_and_metadata.image;
        let rgb_desc = image.get_channel_desc(&["R", "G", "B"]);
        if rgb_desc.is_none() {
            error!(
                @file
                filename.display(),
                "normal map should have rgb channels.",
            );
        }
        let rgb_desc = rgb_desc.unwrap();
        if rgb_desc.size() != 3 {
            error!(
                @file
                filename.display(),
                "normal map should have rgb channels.",
            );
        }
        let image = Arc::new(image);
        self.normal_maps.insert(normal_map_filename, image);

        Ok(())
    }

    pub fn create_media(&self) -> HashMap<String, Arc<Medium>> {
        // TODO: Find a way to partially move out of self.media instead of cloning
        self.media.clone()
    }

    pub fn create_textures(
        &mut self,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        string_interner: &StringInterner<DefaultBackend>,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> ParseResult<NamedTextures> {
        // TODO: Note that albedo spectrum and float textures were created
        //  earlier; if we switch to asynch, we will want to resolve them here.

        for tex in &self.async_spectrum_textures {
            let render_from_texture = tex.1.render_from_object;

            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            let unbounded_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Unbounded,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            let illum_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Illuminant,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            self.textures
                .unbounded_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(unbounded_tex));
            self.textures
                .illuminant_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(illum_tex));
        }

        for tex in &self.serial_float_textures {
            let render_from_texture = tex.1.render_from_object;

            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            // TODO: Will need to pass self.textures to create() functions, so they can resolve textures.
            // Not encessary right now as we only have the FloatConstant texture.
            let float_texture = FloatTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                &tex.1.base.loc,
                &self.textures,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            self.textures
                .float_textures
                .insert(tex.0.to_owned(), Arc::new(float_texture));
        }

        for tex in &self.serial_spectrum_textures {
            let render_from_texture = tex.1.render_from_object;

            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            // TODO: Will need to pass self.textures to create() functions, so they can resolve textures.
            // Not encessary right now as we only have the ConstantSpectrum texture.
            let albedo_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Albedo,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            let unbounded_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Unbounded,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            let illum_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Illuminant,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            )?;

            self.textures
                .albedo_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(albedo_tex));
            self.textures
                .unbounded_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(unbounded_tex));
            self.textures
                .illuminant_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(illum_tex));
        }

        // TODO: It would probably be better to not have to clone the textures here.
        //  Can we return a reference?
        //  Storing self.textures as Arc or Rc doesn't work since we need it mutable.
        //  This is fine for now.
        Ok(self.textures.clone())
    }

    #[allow(clippy::type_complexity)]
    pub fn create_lights(
        &mut self,
        textures: &NamedTextures,
        string_interner: &StringInterner<DefaultBackend>,
        options: &Options,
    ) -> ParseResult<(Arc<[Arc<Light>]>, HashMap<usize, Vec<Arc<Light>>>)> {
        let find_medium = |s: &str, loc: &FileLoc| -> ParseResult<Option<&Arc<Medium>>> {
            if s.is_empty() {
                return Ok(None);
            }

            match self.media.get(s) {
                Some(m) => Ok(Some(m)),
                None => { error!(loc, "medium '{}' not defined", s); },
            }
        };

        let mut shape_index_to_area_lights = HashMap::new();
        // TODO: We'll want to handle alpha textures, but hold off for now.

        let mut lights = Vec::new();

        for i in 0..self.shapes.len() {
            let shape = &mut self.shapes[i];

            if shape.light_index == -1 {
                continue;
            }

            let material_name = if let CurrentGraphicsMaterial::NamedMaterial(material_name) = &shape.material {
                let Some(mut material) = self.named_materials.iter_mut().find(|m| m.0 == *material_name) else {
                    error!(
                        shape.base.loc,
                        "couldn't find named material '{}'",
                        material_name
                    );
                };
                assert!(
                    !material.1.parameters.get_one_string("type", "")?.is_empty()
                );
                material.1.parameters.get_one_string("type", "")?
            } else {
                let CurrentGraphicsMaterial::MaterialIndex(material_index) = shape.material else { unreachable!() };
                assert!(
                    material_index >= 0
                        && (material_index as usize) < self.materials.len()
                );
                string_interner
                    .resolve(self.materials[material_index as usize].name)
                    .unwrap()
                    .to_owned()
            };

            if material_name == "interface" || material_name == "none" || material_name.is_empty() {
                warn!(
                    shape.base.loc,
                    "ignoring area light specification for shape with interface material",
                );
                continue;
            }

            let shape_objects = Shape::create(
                string_interner.resolve(shape.base.name).unwrap(),
                shape.render_from_object,
                shape.object_from_render,
                shape.reverse_orientation,
                &mut shape.base.parameters,
                &textures.float_textures,
                &shape.base.loc,
                options,
            )?;

            // TODO: Support an alpha texture if parameters.get_texture("alpha") is specified.
            let alpha = shape.base.parameters.get_one_float("alpha", 1.0)?;
            let alpha = Arc::new(FloatTexture::Constant(FloatConstantTexture::new(alpha)));

            let mi = Arc::new(MediumInterface::new(
                find_medium(&shape.inside_medium, &shape.base.loc)?.cloned(),
                find_medium(&shape.outside_medium, &shape.base.loc)?.cloned(),
            ));

            let mut shape_lights = Vec::new();
            let area_light_entity = &mut self.area_lights[shape.light_index as usize];
            for ps in shape_objects.iter() {
                let area = Arc::new(Light::create_area(
                    string_interner.resolve(area_light_entity.name).unwrap(),
                    &mut area_light_entity.parameters,
                    shape.render_from_object,
                    mi.clone(),
                    ps.clone(),
                    alpha.clone(),
                    &area_light_entity.loc,
                    options,
                )?);

                lights.push(area.clone());
                shape_lights.push(area);
            }

            shape_index_to_area_lights.insert(i, shape_lights);
        }

        // TODO: We could create other lights in parallel here;
        //  for now, we are creating them in add_light() in self.lights.
        self.lights.append(&mut lights);

        // TODO: We'd rather move self.lights out rather than an expensive clone.
        //   We can switch to make lights vec in this fn though when we parallelize,
        //   which obviates this issue.
        Ok((self.lights.clone().into(), shape_index_to_area_lights))
    }

    pub fn create_materials(
        &mut self,
        textures: &NamedTextures,
        string_interner: &StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) -> ParseResult<(HashMap<String, Arc<Material>>, Vec<Arc<Material>>)> {
        // TODO: Note that we'd create normal_maps here if/when we parallelize.
        //  For now they're already been loaded into self.normal_maps.

        let mut named_materials_out: HashMap<String, Arc<Material>> = HashMap::new();
        for (name, material) in &mut self.named_materials {
            if named_materials_out.iter().any(|nm| nm.0 == name) {
                error!(material.loc, "named material '{}' redefined", name);
            }

            let ty = material.parameters.get_one_string("type", "")?;
            if ty.is_empty() {
                error!(material.loc, "no type specified for material '{}'", name);
            }

            let filename = resolve_filename(
                options,
                &material
                    .parameters
                    .get_one_string("normalmap", "")?,
            );
            let normal_map = if filename.is_empty() {
                None
            } else {
                let image = self.normal_maps.get(&filename);
                if image.is_none() {
                    error!(material.loc, "normal map \"{}\" not found.", filename);
                }
                Some(image.unwrap().clone())
            };

            let mut tex_dict = TextureParameterDictionary::new(material.parameters.clone());
            let m = Arc::new(Material::create(
                &ty,
                &mut tex_dict,
                textures,
                normal_map,
                &named_materials_out,
                cached_spectra,
                &material.loc,
            )?);
            named_materials_out.insert(name.to_string(), m);
        }

        let mut materials_out = Vec::with_capacity(self.materials.len());
        for mtl in &mut self.materials {
            let filename = resolve_filename(
                options,
                &mtl.parameters.get_one_string("normalmap", "")?,
            );
            let normal_map = if filename.is_empty() {
                None
            } else {
                let image = self.normal_maps.get(&filename);
                if image.is_none() {
                    error!(mtl.loc, "normal map \"{}\" not found.", filename);
                }
                Some(image.unwrap().clone())
            };

            let mut tex_dict = TextureParameterDictionary::new(mtl.parameters.clone());
            let m = Arc::new(Material::create(
                string_interner.resolve(mtl.name).unwrap(),
                &mut tex_dict,
                textures,
                normal_map,
                &named_materials_out,
                cached_spectra,
                &mtl.loc,
            )?);
            materials_out.push(m);
        }

        Ok((named_materials_out, materials_out))
    }

    pub fn create_aggregate(
        &mut self,
        textures: &NamedTextures,
        shape_index_to_area_lights: &HashMap<usize, Vec<Arc<Light>>>,
        media: &HashMap<String, Arc<Medium>>,
        named_materials: &HashMap<String, Arc<Material>>,
        materials: &[Arc<Material>],
        string_interner: &StringInterner<DefaultBackend>,
        options: &Options,
    ) -> ParseResult<Arc<Primitive>> {
        let find_medium = |s: &str, loc: &FileLoc| -> ParseResult<Option<&Arc<Medium>>> {
            if s.is_empty() {
                return Ok(None);
            }
            
            match media.get(s) {
                Some(m) => Ok(Some(m)),
                None => { error!(loc, "medium '{}' not defined", s); },
            }
        };

        // TODO: We'll need closure for get_alpha_texture.

        let create_primitives_for_shapes =
            |shapes: &mut [ShapeSceneEntity]| -> ParseResult<Vec<Arc<Primitive>>> {
                let mut shape_vectors: Vec<Vec<Arc<Shape>>> = vec![Vec::new(); shapes.len()];
                // TODO: parallelize
                for i in 0..shapes.len() {
                    let sh = &mut shapes[i];
                    shape_vectors[i] = Shape::create(
                        string_interner.resolve(sh.base.name).unwrap(),
                        sh.render_from_object,
                        sh.object_from_render,
                        sh.reverse_orientation,
                        &mut sh.base.parameters,
                        &textures.float_textures,
                        &sh.base.loc,
                        options,
                    )?;
                }

                let mut primitives = Vec::new();
                for i in 0..shapes.len() {
                    let sh = &mut shapes[i];
                    let shapes = &shape_vectors[i];
                    if shapes.is_empty() {
                        continue;
                    }

                    // TODO: get alpha texture here

                    let mtl = if let CurrentGraphicsMaterial::NamedMaterial(material_name) = &sh.material {
                        named_materials
                            .get(material_name.as_str())
                            .ok_or(error!(@create sh.base.loc, "material '{}' undefined", material_name))?
                    } else {
                        let CurrentGraphicsMaterial::MaterialIndex(material_index) = sh.material else { unreachable!() };
                        assert!(
                            material_index >= 0
                                && (material_index as usize) < materials.len()
                        );
                        &materials[material_index as usize]
                    };

                    let mi = Arc::new(MediumInterface::new(
                        find_medium(&sh.inside_medium, &sh.base.loc)?.cloned(),
                        find_medium(&sh.outside_medium, &sh.base.loc)?.cloned(),
                    ));

                    let area_lights = shape_index_to_area_lights.get(&i);
                    for j in 0..shapes.len() {
                        let area = if sh.light_index != -1 && area_lights.is_some() {
                            let area_light = area_lights.unwrap();
                            Some(area_light[j].clone())
                        } else {
                            None
                        };

                        // TODO: Also check against alpha_tex.is_none()
                        if area.is_none() && !mi.is_transition() {
                            let prim = Arc::new(Primitive::Simple(SimplePrimitive {
                                shape: shapes[j].clone(),
                                material: mtl.clone(),
                            }));
                            primitives.push(prim);
                        } else {
                            let prim = Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                                shapes[j].clone(),
                                mtl.clone(),
                                area,
                                mi.clone(),
                            )));
                            primitives.push(prim);
                        }
                    }
                }
                Ok(primitives)
            };

        log!("Creating shapes...");
        let mut primitives = create_primitives_for_shapes(&mut self.shapes)?;

        self.shapes.clear();
        self.shapes.shrink_to_fit();

        // TODO: Animated shapes, when added.
        clear_log!();

        log!("Creating instances...");
        // TODO: Can we use a SymbolU32 here for the key instead of String?
        let mut instance_definitions: HashMap<String, Option<Arc<Primitive>>> = HashMap::new();
        for inst in &mut self.instance_definitions {
            let instance_primitives = create_primitives_for_shapes(&mut inst.shapes)?;
            // TODO: animated instance primitives

            let instance_primitives = if instance_primitives.len() > 1 {
                let bvh = BvhAggregate::new(
                    instance_primitives,
                    1,
                    BvhSplitMethod::SAH,
                );
                vec![Arc::new(Primitive::BvhAggregate(bvh))]
            } else {
                instance_primitives
            };

            if instance_primitives.is_empty() {
                instance_definitions
                    .insert(string_interner.resolve(inst.name).unwrap().to_owned(), None);
            } else {
                instance_definitions.insert(
                    string_interner.resolve(inst.name).unwrap().to_owned(),
                    Some(instance_primitives[0].clone()),
                );
            }
        }

        self.instance_definitions.clear();
        self.instance_definitions.shrink_to_fit();

        for inst in &self.instances {
            let e = error!(@create @noloc "unknown instance name");
            let instance = instance_definitions
                .get(string_interner.resolve(inst.name).unwrap())
                .ok_or(error!(@create @noloc "unknown instance name"))?;

            if instance.is_none() {
                continue;
            }

            let instance = instance.as_ref().unwrap();

            // TODO: Handle animated instances
            let prim = Arc::new(Primitive::Transformed(TransformedPrimitive::new(
                instance.clone(),
                inst.render_from_instance,
            )));
            primitives.push(prim);
        }

        self.instances.clear();
        self.instances.shrink_to_fit();

        clear_log!();

        log!("Creating top-level accelerator...");
        let aggregate = Arc::new(create_accelerator(
            string_interner
                .resolve(self.accelerator.as_ref().unwrap().name)
                .unwrap(),
            primitives,
            &mut self.accelerator.as_mut().unwrap().parameters,
        )?);
        clear_log!();

        Ok(aggregate)
    }

    pub fn create_integrator(
        &mut self,
        camera: Camera,
        sampler: Sampler,
        accelerator: Arc<Primitive>,
        lights: Arc<[Arc<Light>]>,
        string_interner: &StringInterner<DefaultBackend>,
    ) -> ParseResult<Integrator> {
        Integrator::create(
            string_interner
                .resolve(self.integrator.as_ref().unwrap().name)
                .unwrap(),
            &mut self.integrator.as_mut().unwrap().parameters,
            camera,
            sampler,
            accelerator,
            lights,
            self.film_color_space.as_ref().unwrap().clone(),
        )
    }
}


#[derive(Debug, Clone)]
pub struct SceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    parameters: ParameterDictionary,
}

impl SceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner<DefaultBackend>,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            parameters,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShapeSceneEntity {
    base: SceneEntity,
    render_from_object: Transform,
    object_from_render: Transform,
    reverse_orientation: bool,
    material: CurrentGraphicsMaterial,
    light_index: i32,
    inside_medium: String,
    outside_medium: String,
}

#[derive(Debug, Clone)]
pub struct CameraSceneEntity {
    base: SceneEntity,
    camera_transform: CameraTransform,
    medium: String,
}

#[derive(Debug, Clone)]
pub struct TransformedSceneEntity {
    base: SceneEntity,
    render_from_object: Transform, // TODO: This may need to be AnimatedTransform
}

impl TransformedSceneEntity {
    pub fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        render_from_object: Transform,
    ) -> Self {
        Self {
            base: SceneEntity::new(name, loc, parameters, string_interner),
            render_from_object,
        }
    }
}

pub type MediumSceneEntity = TransformedSceneEntity;
pub type TextureSceneEntity = TransformedSceneEntity;

#[derive(Debug, Clone)]
pub struct LightSceneEntity {
    base: TransformedSceneEntity,
    medium: String,
}

impl LightSceneEntity {
    pub fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        render_from_light: Transform,
        medium: &str
    ) -> LightSceneEntity {
        let base = TransformedSceneEntity::new(name, parameters, string_interner, loc, render_from_light);
        LightSceneEntity {
            base,
            medium: medium.to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstanceSceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    render_from_instance: Transform, // TODO: Might need to be AnimatedTransform
}

impl InstanceSceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        string_interner: &mut StringInterner<DefaultBackend>,
        render_from_instance: Transform,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            render_from_instance,
        }
    }
}

#[derive(Debug)]
pub struct InstanceDefinitionSceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    shapes: Vec<ShapeSceneEntity>,
    // TODO: animated_shapes: Vec<AnimatedShapeSceneEntity>,
}

impl InstanceDefinitionSceneEntity {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            shapes: Vec::new(),
        }
    }
}

const MAX_TRANSFORMS: usize = 2;

#[derive(Debug, Copy, Clone, PartialEq)]
struct TransformSet {
    t: [Transform; MAX_TRANSFORMS],
}

impl TransformSet {
    fn inverse(&self) -> TransformSet {
        let mut t_inv = TransformSet::default();
        for i in 0..MAX_TRANSFORMS {
            t_inv.t[i] = Transform::inverse(&self.t[i]);
        }
        t_inv
    }

    fn is_animated(&self) -> bool {
        for i in 0..(MAX_TRANSFORMS - 1) {
            if self.t[i] != self.t[i + 1] {
                return true;
            }
        }

        false
    }
}

impl Default for TransformSet {
    fn default() -> Self {
        Self {
            t: [Transform::default(); MAX_TRANSFORMS],
        }
    }
}

impl Index<usize> for TransformSet {
    type Output = Transform;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &self.t[index]
    }
}

impl IndexMut<usize> for TransformSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &mut self.t[index]
    }
}

#[derive(Debug, Clone)]
enum CurrentGraphicsMaterial {
    MaterialIndex(i32),
    NamedMaterial(String),
}

#[derive(Debug, Clone)]
struct GraphicsState {
    current_inside_medium: String,
    current_outside_medium: String,
    current_material: CurrentGraphicsMaterial,

    area_light_name: String,
    area_light_params: ParameterDictionary,
    area_light_loc: FileLoc,

    shape_attributes: ParsedParameterVector,
    light_attributes: ParsedParameterVector,
    material_attributes: ParsedParameterVector,
    medium_attributes: ParsedParameterVector,
    texture_attributes: ParsedParameterVector,
    reverse_orientation: bool,
    color_space: Arc<RgbColorSpace>,
    ctm: TransformSet,
    active_transform_bits: u32,
    transform_start_time: Float,
    transform_end_time: Float,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            current_inside_medium: Default::default(),
            current_outside_medium: Default::default(),
            current_material: CurrentGraphicsMaterial::MaterialIndex(0),
            area_light_name: Default::default(),
            area_light_params: Default::default(),
            area_light_loc: Default::default(),
            shape_attributes: Default::default(),
            light_attributes: Default::default(),
            material_attributes: Default::default(),
            medium_attributes: Default::default(),
            texture_attributes: Default::default(),
            reverse_orientation: Default::default(),
            color_space: RgbColorSpace::get_named(NamedColorSpace::SRgb).clone(),
            ctm: Default::default(),
            active_transform_bits: BasicSceneBuilder::ALL_TRANSFORM_BITS,
            transform_start_time: 0.0,
            transform_end_time: 1.0,
        }
    }
}

impl GraphicsState {
    pub fn for_active_transforms(&mut self, f: impl Fn(&mut Transform) -> ParseResult<()>) -> ParseResult<()> {
        for i in 0..MAX_TRANSFORMS {
            if self.active_transform_bits & (1 << i) != 0 {
                f(&mut self.ctm[i])?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum BlockState {
    OptionsBlock,
    WorldBlock,
}

struct ActiveInstanceDefinition {
    pub active_imports: AtomicI32,
    pub entity: InstanceDefinitionSceneEntity,
    pub parent: Option<Arc<ActiveInstanceDefinition>>,
}

impl ActiveInstanceDefinition {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> Self {
        Self {
            active_imports: AtomicI32::new(1),
            entity: InstanceDefinitionSceneEntity::new(name, loc, string_interner),
            parent: None,
        }
    }
}

pub struct BasicSceneBuilder {
    scene: Box<BasicScene>,
    current_block: BlockState,
    graphics_state: GraphicsState,
    named_coordinate_systems: HashMap<String, TransformSet>,
    render_from_world: Transform,
    // TODO: Transform cache
    pushed_graphics_states: Vec<GraphicsState>,
    push_stack: Vec<(u8, FileLoc)>,
    // TODO: Instance definition members
    
    shapes: Vec<ShapeSceneEntity>,
    instance_uses: Vec<InstanceSceneEntity>,

    named_material_names: HashSet<String>,
    named_medium_names: HashSet<String>,
    float_texture_names: HashSet<String>,
    spectrum_texture_names: HashSet<String>,
    instance_names: HashSet<String>,

    current_material_index: i32,
    sampler: SceneEntity,
    film: SceneEntity,
    integrator: SceneEntity,
    filter: SceneEntity,
    accelerator: SceneEntity,
    camera: CameraSceneEntity,

    active_instance_definition: Option<ActiveInstanceDefinition>,
}

impl BasicSceneBuilder {
    const START_TRANSFORM_BITS: u32 = 1 << 0;
    const END_TRANSFORM_BITS: u32 = 1 << 1;
    const ALL_TRANSFORM_BITS: u32 = (1 << MAX_TRANSFORMS) - 1;

    pub fn new(
        scene: Box<BasicScene>,
        string_interner: &mut StringInterner<DefaultBackend>,
        options: &Options,
    ) -> BasicSceneBuilder {
        let sampler = SceneEntity {
            name: string_interner.get_or_intern("zsobol"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::default(),
        };

        let film = SceneEntity {
            name: string_interner.get_or_intern("rgb"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::new(
                ParsedParameterVector::new(),
                RgbColorSpace::get_named(NamedColorSpace::SRgb).clone(),
            ),
        };

        let integrator = SceneEntity {
            name: string_interner.get_or_intern("volpath"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::default(),
        };

        let filter = SceneEntity {
            name: string_interner.get_or_intern("gaussian"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::default(),
        };

        let accelerator = SceneEntity {
            name: string_interner.get_or_intern("bvh"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::default(),
        };

        let camera = CameraSceneEntity {
            base: SceneEntity {
                name: string_interner.get_or_intern("perspective"),
                loc: FileLoc::default(),
                parameters: ParameterDictionary::default(),
            },
            camera_transform: CameraTransform::default(),
            medium: String::new(),
        };

        let mut builder = BasicSceneBuilder {
            scene,
            current_block: BlockState::OptionsBlock,
            graphics_state: GraphicsState::default(),
            named_coordinate_systems: HashMap::new(),
            render_from_world: Transform::default(),
            pushed_graphics_states: Vec::new(),
            push_stack: Vec::new(),
            shapes: Vec::new(),
            instance_uses: Vec::new(),
            named_material_names: HashSet::new(),
            named_medium_names: HashSet::new(),
            float_texture_names: HashSet::new(),
            spectrum_texture_names: HashSet::new(),
            instance_names: HashSet::new(),
            current_material_index: 0,
            sampler,
            film,
            integrator,
            filter,
            accelerator,
            camera,
            active_instance_definition: None,
        };

        let dict = ParameterDictionary::new(
            ParsedParameterVector::new(),
            RgbColorSpace::get_named(NamedColorSpace::SRgb).clone(),
        );

        let diffuse = SceneEntity::new("diffuse", FileLoc::default(), dict, string_interner);
        builder.current_material_index = builder.scene.add_material(diffuse, options);

        builder
    }

    pub fn done(self) -> Box<BasicScene> {
        self.scene
    }

    fn render_from_object(&self) -> Transform {
        // TODO: Support AnimatedTransform
        self.render_from_world.apply(self.graphics_state.ctm[0])
    }
    
    fn ctm_is_animated(&self) -> bool {
        self.graphics_state.ctm.is_animated()
    }
}

impl ParserTarget for BasicSceneBuilder {
    fn shape(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        // TODO: Verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.shape_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        let area_light_index = if self.graphics_state.area_light_name.is_empty() {
            -1
        } else {
            if self.active_instance_definition.is_some() {
                warn!(loc, "area lights not supported with object instancing");
            }
            
            self.scene.add_area_light(SceneEntity::new(
                &self.graphics_state.area_light_name,
                self.graphics_state.area_light_loc.clone(),
                self.graphics_state.area_light_params.clone(),
                string_interner,
            ))
        };

        if self.ctm_is_animated() {
            todo!("Support AnimatedTransform")
        } else {
            // TODO: Use transform cache
            let render_from_object = self.render_from_object();
            let object_from_render = render_from_object.inverse();

            let entity = ShapeSceneEntity {
                base: SceneEntity::new(name, loc, dict, string_interner),
                render_from_object,
                object_from_render,
                reverse_orientation: self.graphics_state.reverse_orientation,
                material: self.graphics_state.current_material.clone(),
                light_index: area_light_index,
                inside_medium: self.graphics_state.current_inside_medium.clone(),
                outside_medium: self.graphics_state.current_outside_medium.clone(),
            };

            if let Some(active_instance_definition) = &mut self.active_instance_definition {
                active_instance_definition.entity.shapes.push(entity)
            } else {
                self.shapes.push(entity)
            }
        }
    }

    fn option(&mut self, name: &str, value: &str, options: &mut Options, loc: FileLoc) -> ParseResult<()> {
        let name = normalize_arg(name);

        match name.as_str() {
            "disablepixeljitter" => match value {
                "true" => options.disable_pixel_jitter = true,
                "false" => options.disable_pixel_jitter = false,
                _ => { error!(loc, "unknown option value '{}'", value); },
            },
            "disabletexturefiltering" => match value {
                "true" => options.disable_texture_filtering = true,
                "false" => options.disable_texture_filtering = false,
                _ => { error!(loc, "unknown option value '{}'", value); },
            },
            "disablewavelengthjitter" => match value {
                "true" => options.disable_wavelength_jitter = true,
                "false" => options.disable_wavelength_jitter = false,
                _ => { error!(loc, "unknown option_value '{}'", value); },
            },
            "displacementedgescale" => {
                options.displacement_edge_scale = value.parse()
                    .map_err(|_| error!(@create loc, "unable to parse option value '{}'", value))?;
            },
            "msereferenceimage" => {
                if value.len() < 3 || !value.starts_with('\"') || !value.ends_with('\"') {
                    error!(loc, "expected quotes string for option value '{}'", value);
                }
                options.mse_reference_image = Some(value[1..value.len() - 1].to_owned());
            },
            "msereferenceout" => {
                if value.len() < 3 || !value.starts_with('\"') || !value.ends_with('\"') {
                    error!(loc, "expected quotes string for option value '{}'", value);
                }
                options.mse_reference_output = Some(value[1..value.len() - 1].to_owned());
            },
            "rendercoordsys" => {
                if value.len() < 3 || !value.starts_with('\"') || !value.ends_with('\"') {
                    error!(loc, "expected quotes string for option value '{}'", value);
                }
                let render_coord_sys = value[1..value.len() - 1].to_owned();
                match render_coord_sys.as_str() {
                    "camera" => options.rendering_space = CameraRenderingSpace::Camera,
                    "cameraworld" => options.rendering_space = CameraRenderingSpace::CameraWorld,
                    "world" => options.rendering_space = CameraRenderingSpace::World,
                    _ => { error!(loc, "unknown option value '{}'", value); },
                }
            },
            "seed" => {
                options.seed = value.parse()
                    .map_err(|_| error!(@create loc, "unable to parse option value '{}'", value))?;
            },
            "forcediffuse" => match value {
                "true" => options.force_diffuse = true,
                "false" => options.force_diffuse = false,
                _ => { error!(loc, "unknown option value '{}' (expected true|false)", value); },
            },
            "pixelstats" => match value {
                "true" => options.record_pixel_statistics = true,
                "false" => options.record_pixel_statistics = false,
                _ => { error!(loc, "unknown option value '{}' (expected true|false)", value); },
            },
            "wavefront" => match value {
                "true" => options.wavefront = true,
                "false" => options.wavefront = false,
                _ => { error!(loc, "unknown option value '{}' (expected true|false)", value); },
            },
            _ => { error!(loc, "unknown option '{}'", name); },
        }

        Ok(())
    }

    fn identity(&mut self, loc: FileLoc) -> ParseResult<()> {
        self.graphics_state.for_active_transforms(|t: &mut Transform| { *t = Transform::default(); Ok(()) })
    }

    fn translate(&mut self, dx: Float, dy: Float, dz: Float, loc: FileLoc) -> ParseResult<()> {
        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            *t = t.apply(Transform::from_translation(Point3f::new(dx, dy, dz)));
            Ok(())
        })
    }
    
    fn scale(&mut self, sx: Float, sy: Float, sz: Float, loc: FileLoc) -> ParseResult<()> {
        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            *t = t.apply(Transform::from_scale(Vec3f::new(sx, sy, sz)));
            Ok(())
        })
    }

    fn rotate(&mut self, mut angle: Float, ax: Float, ay: Float, az: Float, loc: FileLoc) -> ParseResult<()> {
        angle = to_radians(angle);
        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            *t = t.apply(Transform::from_rotation(angle.sin(), angle.cos(), Vec3f::new(ax, ay, az)));
            Ok(())
        })
    }

    fn look_at(
        &mut self,
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
        loc: FileLoc,
    ) -> ParseResult<()> {
        let transform = Transform::looking_at(
            Point3f::new(ex, ey, ez),
            Point3f::new(lx, ly, lz),
            Vec3f::new(ux, uy, uz),
        );

        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            *t = t.apply(transform);
            Ok(())
        })
    }

    fn transform(&mut self, transform: [Float; 16], loc: FileLoc) -> ParseResult<()> {
        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            let m = Mat4::from(transform);
            if let Some(m_inv) = m.try_inverse() {
                *t = Transform::new(m, m_inv).transpose();
            } else {
                error!(loc, "matrix {:?} is not inversible", transform);
            }
            
            Ok(())
        })
    }

    fn concat_transform(&mut self, transform: [Float; 16], loc: FileLoc) -> ParseResult<()> {
        self.graphics_state.for_active_transforms(|t: &mut Transform| {
            let m = Mat4::from(transform);
            if let Some(m_inv) = m.try_inverse() {
                *t = t.apply(Transform::new(m, m_inv).transpose())
            } else {
                error!(loc, "matrix {:?} is not inversible", transform);
            }

            Ok(())
        })
    }

    fn coordinate_system(&mut self, name: &str, loc: FileLoc) {
        // TODO: Normalize name to utf8
        self.named_coordinate_systems.insert(name.to_owned(), self.graphics_state.ctm);
    }

    fn coordinate_sys_transform(&mut self, name: &str, loc: FileLoc) {
        // TODO: Normalize name to utf8
        if let Some(ctm) = self.named_coordinate_systems.get(name) {
            self.graphics_state.ctm = *ctm;
        } else {
            warn!(loc, "couldn't find named coordinate system '{}'", name);
        }
    }

    fn active_transform_all(&mut self, loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
    }

    fn active_transform_end_time(&mut self, loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::END_TRANSFORM_BITS;
    }

    fn active_transform_start_time(&mut self, loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::START_TRANSFORM_BITS;
    }

    fn transform_times(&mut self, start: Float, end: Float, loc: FileLoc) {
        // TODO: verify options
        self.graphics_state.transform_start_time = start;
        self.graphics_state.transform_end_time = end;
    }

    fn color_space(&mut self, n: &str, loc: FileLoc) -> ParseResult<()> {
        let cs = RgbColorSpace::get_named(NamedColorSpace::from_str(n).map_err(|e| e.with_loc(loc))?);
        self.graphics_state.color_space = cs.clone();
        Ok(())
    }

    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO: verify options
        self.filter = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO: verify options
        self.film = SceneEntity::new(film_type, loc, dict, string_interner);
    }

    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO: verify options
        self.accelerator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO: verify options
        self.integrator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn camera(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());

        let camera_from_world = &self.graphics_state.ctm;
        let world_from_camera = camera_from_world.inverse();

        // TODO: AnimatedTransform
        let camera_transform = CameraTransform::new(world_from_camera[0], options);
        self.named_coordinate_systems.insert("camera".to_owned(), world_from_camera);
        self.render_from_world = camera_transform.render_from_world();

        self.camera = CameraSceneEntity {
            base: SceneEntity::new(name, loc, dict, string_interner),
            camera_transform,
            medium: self.graphics_state.current_outside_medium.clone(),
        }
    }

    fn make_named_medium(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: FileLoc,
    ) -> ParseResult<()> {
        // TODO: Normalize name to utf8

        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.medium_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if self.named_medium_names.insert(name.to_owned()) {
            self.scene.add_medium(
                MediumSceneEntity::new(name, dict, string_interner, loc, self.render_from_object()),
                string_interner,
                cached_spectra,
            );
        } else {
            // TODO: defer error instead
            error!(loc, "named medium '{}' redefined", name);
        }

        Ok(())
    }

    fn medium_interface(&mut self, inside_name: &str, outside_name: &str, loc: FileLoc) {
        // TODO: Normalize inside_name + outside_name to utf8

        self.graphics_state.current_inside_medium = inside_name.to_owned();
        self.graphics_state.current_outside_medium = outside_name.to_owned();
    }

    fn sampler(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO: verify options
        self.sampler = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn world_begin(
        &mut self,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    ) {
        self.current_block = BlockState::WorldBlock;
        for i in 0..MAX_TRANSFORMS {
            self.graphics_state.ctm[i] = Transform::default();
        }

        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
        self.named_coordinate_systems.insert("world".into(), self.graphics_state.ctm);

        self.scene.set_options(
            self.filter.clone(),
            self.film.clone(),
            self.camera.clone(),
            self.sampler.clone(),
            self.integrator.clone(),
            self.accelerator.clone(),
            string_interner,
            options
        );
    }

    fn attribute_begin(&mut self, loc: FileLoc) {
        // TODO: verify world
        self.pushed_graphics_states.push(self.graphics_state.clone());
        self.push_stack.push((b'a', loc));
    }

    fn attribute_end(&mut self, loc: FileLoc) -> ParseResult<()> {
        // TODO: verify world
        if self.push_stack.is_empty() {
            error!(loc, "unmatched attribute_end statement");
        }

        // NOTE: Keep the following consistent with code in ObjectEnd
        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == b'o' {
            error!(loc, "mismatched nesting: open ObjectBegin from {} at attribute_end", self.push_stack.last().unwrap().1);
        } else {
            assert!(self.push_stack.last().unwrap().0 == b'a');
        }

        self.push_stack.pop();

        Ok(())
    }

    fn attribute(&mut self, target: &str, mut params: ParsedParameterVector, loc: FileLoc) -> ParseResult<()> {
        let current_attributes = match target {
            "shape" => &mut self.graphics_state.shape_attributes,
            "light" => &mut self.graphics_state.light_attributes,
            "material" => &mut self.graphics_state.material_attributes,
            "medium" => &mut self.graphics_state.medium_attributes,
            "texture" => &mut self.graphics_state.texture_attributes,
            _ => { error!(loc, "unknown attribute target '{}'", target); },
        };

        for p in params.iter_mut() {
            p.may_be_unused = true;
            p.color_space = Some(self.graphics_state.color_space.clone());
            current_attributes.push(p.to_owned());
        }

        Ok(())
    }

    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> ParseResult<()> {
        // TODO: normalize name to utf8
        // TODO: verify world

        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.texture_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if texture_type != "float" && texture_type != "spectrum" {
            error!(loc, "texture type '{}' unknown.", texture_type);
        }

        let names = if texture_type == "float" {
            &mut self.float_texture_names
        } else {
            &mut self.spectrum_texture_names
        };

        if names.insert(name.to_owned()) {
            match texture_type {
                "float" => {
                    self.scene.add_float_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                        string_interner,
                        options,
                        texture_cache,
                        gamma_encoding_cache,
                    )
                },
                "spectrum" => {
                    self.scene.add_spectrum_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                        string_interner,
                        cached_spectra,
                        options,
                        texture_cache,
                        gamma_encoding_cache,
                    )
                },
                _ => { error!(loc, "unknown texture type '{}'", texture_type); },
            }
        } else {
            // TODO: defer error instead
            error!(loc, "texture '{}' redefined", name);
        }
    }

    fn material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    ) {
        // TODO: verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        self.graphics_state.current_material = CurrentGraphicsMaterial::MaterialIndex(
            self.scene.add_material(SceneEntity::new(name, loc, dict, string_interner), options)
        );
    }

    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    ) -> ParseResult<()> {
        // TODO: normalize name to utf8
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if self.named_material_names.insert(name.to_owned()) {
            self.scene.add_named_material(name, SceneEntity::new(name, loc, dict, string_interner), options);
        } else {
            // TODO: defer error instead
            error!(loc, "named material '{}' redefined", name);
        }

        Ok(())
    }

    fn named_material(&mut self, name: &str, loc: FileLoc) {
        // TODO: normalize name to utf8
        // TODO: verify world
        self.graphics_state.current_material = CurrentGraphicsMaterial::NamedMaterial(name.to_owned());
        self.current_material_index = -1;
    }

    fn light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) {
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        self.scene.add_light(
            &mut LightSceneEntity::new(
                name,
                dict,
                string_interner,
                loc,
                self.render_from_object(),
                &self.graphics_state.current_outside_medium,
            ),
            string_interner,
            cached_spectra,
            options,
        );
    }

    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc) {
        // TODO: verify world
        self.graphics_state.area_light_name = name.to_owned();
        self.graphics_state.area_light_params = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );
        self.graphics_state.area_light_loc = loc;
    }

    fn reverse_orientation(&mut self, loc: FileLoc) {
        // TODO: verify world
        self.graphics_state.reverse_orientation = !self.graphics_state.reverse_orientation;
    }

    fn object_begin(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> ParseResult<()> {
        // TODO: verify world
        // TODO: normalize name to utf8

        self.pushed_graphics_states.push(self.graphics_state.clone());
        self.push_stack.push((b'o', loc.clone()));

        if self.active_instance_definition.is_some() {
            error!(loc, "ObjectBegin called inside of instance definition");
        }

        let inserted = self.instance_names.insert(name.to_owned());
        if !inserted {
            error!(loc, "ObjectBegin trying to redefine object instance '{}'", name);
        }

        self.active_instance_definition = Some(ActiveInstanceDefinition::new(name, loc, string_interner));

        Ok(())
    }

    fn object_end(&mut self, loc: FileLoc) -> ParseResult<()> {
        // TODO: verify world
        if self.active_instance_definition.is_none() {
            error!(loc, "ObjectEnd called outside an instance definition");
        }

        if self.active_instance_definition.as_ref().unwrap().parent.is_some() {
            error!(loc, "ObjectEnd called inside import for instance definition");
        }

        // NOTE: Keep the following consistent with AttributeEnd
        if self.pushed_graphics_states.last().is_none() {
            error!(loc, "unmatched ObjectEnd statement");
        }

        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == b'a' {
            error!(loc, "mismatched nesting: open AttributeBegin from {} at ObjectEnd", self.push_stack.last().unwrap().1);
        } else {
            assert!(self.push_stack.last().unwrap().0 == b'o');
        }

        self.push_stack.pop();

        let active_instance_definition = self.active_instance_definition.take().unwrap();
        active_instance_definition.active_imports.fetch_sub(1, Ordering::SeqCst);

        if active_instance_definition.active_imports.load(Ordering::SeqCst) == 0 {
            self.scene.add_instance_definition(active_instance_definition.entity);
        }

        self.active_instance_definition = None;

        Ok(())
    }

    fn object_instance(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> ParseResult<()> {
        // TODO: normalize name to utf8
        // TODO: verify world

        if self.active_instance_definition.is_some() {
            error!(loc, "ObjectInstance called inside of instance definition");
        }

        let world_from_render = self.render_from_world.inverse();

        if self.ctm_is_animated() {
            todo!("Support AnimatedTransform")
        }

        // TODO: use transform cache
        let render_from_instance = self.render_from_object().apply(world_from_render);
        let entity = InstanceSceneEntity::new(name, loc, string_interner, render_from_instance);
        self.instance_uses.push(entity);

        Ok(())
    }

    fn end_of_files(&mut self) -> ParseResult<()> {
        if self.current_block != BlockState::WorldBlock {
            error!(@noloc "End of files before WorldBegin");
        }

        if !self.pushed_graphics_states.is_empty() {
            error!(@noloc "Missing end to AttributeBegin");
        }

        // TODO: when defered errors are implemented, check for them here.

        if !self.shapes.is_empty() {
            self.scene.add_shapes(&self.shapes);
        }
        if !self.instance_uses.is_empty() {
            self.scene.add_instance_uses(&self.instance_uses);
        }

        self.scene.done();

        Ok(())
    }
}
