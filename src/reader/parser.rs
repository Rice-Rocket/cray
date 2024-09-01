use std::{collections::HashMap, env, fs::{self, File}, io::{BufReader, Read}, path::Path, slice, sync::{Arc, Mutex}};

use flate2::bufread::GzDecoder;
use string_interner::{DefaultBackend, StringInterner};

use crate::{clear_log, color::{rgb_xyz::ColorEncodingCache, spectrum::Spectrum}, file::set_search_directory, log, mipmap::MIPMap, options::Options, texture::TexInfo, Float};

use super::{error::{Error, Result}, param::{Param, ParamList}, target::{FileLoc, ParserTarget}, token::{Directive, Token}, tokenizer::Tokenizer};

pub fn parse_files<T: ParserTarget>(
    files: &[&str],
    target: &mut T,
    options: &mut Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
) {
    if files.is_empty() {
        todo!("read from stdin when no files are provided")
    }

    for file in files {
        if *file != "-" {
            set_search_directory(options, file);
        }

        let data = fs::read_to_string(file).unwrap();
        parse_str(&data, target, options, string_interner, cached_spectra, texture_cache, gamma_encoding_cache);
    }
}

pub fn parse_str<T: ParserTarget>(
    data: &str,
    target: &mut T,
    options: &mut Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
) {
    let mut parsers = Vec::new();
    parsers.push(Parser::new(data));

    let mut includes = Vec::new();

    while let Some(parser) = parsers.last_mut() {
        let element = match parser.parse_next() {
            Ok(element) => element,
            Err(Error::EndOfFile) => {
                clear_log!();
                parsers.pop();
                continue;
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        };

        // TODO: Track FileLoc in tokenizer and pass it in here
        let loc = FileLoc::default();
        match element {
            Element::Include(path_name) => {
                let path = Path::new(path_name);
                let full_path;

                let path = if path.is_absolute() {
                    path
                } else {
                    full_path = match &options.search_directory {
                        Some(directory) => directory.join(path),
                        None => env::current_dir().unwrap().join(path),
                    };

                    full_path.as_path()
                };

                let f = File::open(path).unwrap();
                let mut bufreader = BufReader::new(f);

                let parser = if path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| ext.ends_with("gz")) {
                    log!("Decoding included scene file '{}'", path_name);

                    let mut decoder = GzDecoder::new(bufreader);
                    let mut s = String::new();
                    decoder.read_to_string(&mut s).unwrap();

                    let raw = s.as_bytes();
                    let raw_ptr = raw.as_ptr();
                    let raw_len = raw.len();

                    includes.push(s);

                    clear_log!();

                    Parser::new(unsafe {
                        let byte_slice = slice::from_raw_parts(raw_ptr, raw_len);
                        std::str::from_utf8_unchecked(byte_slice)
                    })
                } else {
                    let mut s = String::new();
                    bufreader.read_to_string(&mut s).unwrap();

                    let raw = s.as_bytes();
                    let raw_ptr = raw.as_ptr();
                    let raw_len = raw.len();
                    
                    includes.push(s);

                    Parser::new(unsafe {
                        let byte_slice = slice::from_raw_parts(raw_ptr, raw_len);
                        std::str::from_utf8_unchecked(byte_slice)
                    })
                };

                log!("Parsing included scene file '{}'", path_name);

                parsers.push(parser);
            },
            Element::Import(_) => todo!("Support import directive"),
            Element::Option(param) => target.option(param.name, param.value, options, loc),
            Element::Film { ty, params } => target.film(ty, params.into(), string_interner, loc),
            Element::ColorSpace { ty } => target.color_space(ty, loc),
            Element::Camera { ty, params } => target.camera(ty, params.into(), string_interner, loc, options),
            Element::Sampler { ty, params } => target.sampler(ty, params.into(), string_interner, loc),
            Element::Integrator { ty, params } => target.integrator(ty, params.into(), string_interner, loc),
            Element::Accelerator { ty, params } => target.accelerator(ty, params.into(), string_interner, loc),
            Element::CoordinateSystem { name } => target.coordinate_system(name, loc),
            Element::CoordSysTransform { name } => target.coordinate_sys_transform(name, loc),
            Element::PixelFilter { name, params } => target.pixel_filter(name, params.into(), string_interner, loc),
            Element::Identity => target.identity(loc),
            Element::Translate { v } => target.translate(v[0], v[1], v[2], loc),
            Element::Scale { v } => target.scale(v[0], v[1], v[2], loc),
            Element::Rotate { angle, v } => target.rotate(angle, v[0], v[1], v[2], loc),
            Element::LookAt { eye, look_at, up } => target.look_at(
                eye[0], eye[1], eye[2], look_at[0], look_at[1], look_at[2],
                up[0], up[1], up[2], loc,
            ),
            Element::Transform { m } => target.transform(m, loc),
            Element::ConcatTransform { m } => target.concat_transform(m, loc),
            Element::TransformTimes { start, end } => target.transform_times(start, end, loc),
            Element::ActiveTransform { ty } => match ty {
                "StartTime" => target.active_transform_start_time(loc),
                "EndTime" => target.active_transform_end_time(loc),
                "All" => target.active_transform_all(loc),
                _ => todo!("Unknown active transform type"),
            },
            Element::ReverseOrientation => target.reverse_orientation(loc),
            Element::WorldBegin => target.world_begin(string_interner, loc, options),
            Element::AttributeBegin => target.attribute_begin(loc),
            Element::AttributeEnd => target.attribute_end(loc),
            Element::Attribute { attr_target, params } => target.attribute(attr_target, params.into(), loc),
            Element::LightSource { ty, params } => target.light_source(
                ty, params.into(), string_interner, loc,
                cached_spectra, options,
            ),
            Element::AreaLightSource { ty, params } => target.area_light_source(ty, params.into(), loc),
            Element::Material { ty, params } => target.material(ty, params.into(), string_interner, loc, options),
            Element::MakeNamedMaterial { name, params } => target.make_named_material(name, params.into(), string_interner, loc, options),
            Element::NamedMaterial { name } => target.named_material(name, loc),
            Element::Texture { name, ty, class, params } => target.texture(
                name, ty, class, params.into(), string_interner, loc,
                options, cached_spectra, texture_cache, gamma_encoding_cache,
            ),
            Element::Shape { name, params } => target.shape(name, params.into(), string_interner, loc),
            Element::ObjectBegin { name } => target.object_begin(name, loc, string_interner),
            Element::ObjectEnd => target.object_end(loc),
            Element::ObjectInstance { name } => target.object_instance(name, loc, string_interner),
            Element::MakeNamedMedium { name, params } => target.make_named_medium(name, params.into(), string_interner, cached_spectra, loc),
            Element::MediumInterface { interior, exterior } => target.medium_interface(interior, exterior, loc),
        }
    }

    target.end_of_files()
}

#[derive(Debug, PartialEq)]
pub enum Element<'a> {
    /// Include behaves similarly to the #include directive in C++: parsing of the current file is suspended,
    /// the specified file is parsed in its entirety, and only then does parsing of the current file resume.
    /// Its effect is equivalent to direct text substitution of the included file.
    Include(&'a str),
    Import(&'a str),
    Option(Param<'a>),
    Film {
        ty: &'a str,
        params: ParamList<'a>,
    },
    ColorSpace {
        ty: &'a str,
    },
    Camera {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Sampler {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Integrator {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Accelerator {
        ty: &'a str,
        params: ParamList<'a>,
    },
    CoordinateSystem {
        name: &'a str,
    },
    CoordSysTransform {
        name: &'a str,
    },
    PixelFilter {
        name: &'a str,
        params: ParamList<'a>,
    },
    Identity,
    /// `Translate x y z`
    Translate {
        v: [Float; 3],
    },
    /// `Scale x y z`
    Scale {
        v: [Float; 3],
    },
    /// `Rotate angle x y z`
    Rotate {
        angle: Float,
        v: [Float; 3],
    },
    /// `LookAt eye_x eye_y eye_z look_x look_y look_z up_x up_y up_z`
    LookAt {
        eye: [Float; 3],
        look_at: [Float; 3],
        up: [Float; 3],
    },
    /// `Transform m00 ... m33`
    Transform {
        m: [Float; 16],
    },
    /// `ConcatTransform m00 .. m33`
    ConcatTransform {
        m: [Float; 16],
    },
    /// `TransformTimes start end`.
    TransformTimes {
        start: Float,
        end: Float,
    },
    ActiveTransform {
        ty: &'a str,
    },
    /// `ReverseOrientation`.
    ReverseOrientation,
    /// `WorldBegin`
    WorldBegin,
    /// `AttributeBegin`
    AttributeBegin,
    /// `AttributeEnd`
    AttributeEnd,
    /// `Attribute "target" parameter-list`
    Attribute {
        attr_target: &'a str,
        params: ParamList<'a>,
    },
    LightSource {
        ty: &'a str,
        params: ParamList<'a>,
    },
    AreaLightSource {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Material {
        ty: &'a str,
        params: ParamList<'a>,
    },
    MakeNamedMaterial {
        name: &'a str,
        params: ParamList<'a>,
    },
    NamedMaterial {
        name: &'a str,
    },
    /// `Texture "name" "type" "class" [ parameter-list ]`
    Texture {
        name: &'a str,
        ty: &'a str,
        class: &'a str,
        params: ParamList<'a>,
    },
    /// `Shape "name" parameter-list`
    Shape {
        name: &'a str,
        params: ParamList<'a>,
    },
    ObjectBegin {
        name: &'a str,
    },
    ObjectEnd,
    ObjectInstance {
        name: &'a str,
    },
    MakeNamedMedium {
        name: &'a str,
        params: ParamList<'a>,
    },
    MediumInterface {
        interior: &'a str,
        exterior: &'a str,
    },
}

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(str: &'a str) -> Self {
        let tokenizer = Tokenizer::new(str);
        Self { tokenizer }
    }

    pub fn parse_next(&mut self) -> Result<Element<'a>> {
        let Some(next_token) = self.tokenizer.next() else {
            return Err(Error::EndOfFile);
        };

        let directive = next_token.directive().ok_or(Error::UnknownDirective(next_token.value().to_owned()))?;

        let element = match directive {
            Directive::Include => Element::Include(self.read_str()?),
            Directive::Import => Element::Import(self.read_str()?),
            Directive::Option => Element::Option(self.read_param()?),
            Directive::Film => Element::Film { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::ColorSpace => Element::ColorSpace { ty: self.read_str()? },
            Directive::Camera => Element::Camera { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::Sampler => Element::Sampler { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::Integrator => Element::Integrator { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::Accelerator => Element::Accelerator { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::CoordinateSystem => Element::CoordinateSystem { name: self.read_str()? },
            Directive::CoordSysTransform => Element::CoordSysTransform { name: self.read_str()? },
            Directive::PixelFilter => Element::PixelFilter { name: self.read_str()?, params: self.read_param_list()? },
            Directive::Identity => Element::Identity,
            Directive::Translate => Element::Translate { v: self.read_point()? },
            Directive::Scale => Element::Scale { v: self.read_point()? },
            Directive::Rotate => Element::Rotate { angle: self.read_float()?, v: self.read_point()? },
            Directive::LookAt => Element::LookAt { eye: self.read_point()?, look_at: self.read_point()?, up: self.read_point()? },
            Directive::Transform => {
                self.skip_brace()?;
                let elem = Element::Transform { m: self.read_matrix()? };
                self.skip_brace()?;
                elem
            },
            Directive::ConcatTransform => {
                self.skip_brace()?;
                let elem = Element::ConcatTransform { m: self.read_matrix()? };
                self.skip_brace()?;
                elem
            },
            Directive::TransformTimes => Element::TransformTimes { start: self.read_float()?, end: self.read_float()? },
            Directive::ActiveTransform => Element::ActiveTransform { ty: self.read_str()? },
            Directive::ReverseOrientation => Element::ReverseOrientation,
            Directive::WorldBegin => Element::WorldBegin,
            Directive::AttributeBegin => Element::AttributeBegin,
            Directive::AttributeEnd => Element::AttributeEnd,
            Directive::Attribute => Element::Attribute { attr_target: self.read_str()?, params: self.read_param_list()? },
            Directive::LightSource => Element::LightSource { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::AreaLightSource => Element::AreaLightSource { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::Material => Element::Material { ty: self.read_str()?, params: self.read_param_list()? },
            Directive::MakeNamedMaterial => Element::MakeNamedMaterial { name: self.read_str()?, params: self.read_param_list()? },
            Directive::NamedMaterial => Element::NamedMaterial { name: self.read_str()? },
            Directive::Texture => Element::Texture {
                name: self.read_str()?,
                ty: self.read_str()?,
                class: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Shape => Element::Shape { name: self.read_str()?, params: self.read_param_list()? },
            Directive::ObjectBegin => Element::ObjectBegin { name: self.read_str()? },
            Directive::ObjectEnd => Element::ObjectEnd,
            Directive::ObjectInstance => Element::ObjectInstance { name: self.read_str()? },
            Directive::MakeNamedMedium => Element::MakeNamedMedium { name: self.read_str()?, params: self.read_param_list()? },
            Directive::MediumInterface => Element::MediumInterface { interior: self.read_str()?, exterior: self.read_str()? },
        };

        Ok(element)
    }

    fn skip_brace(&mut self) -> Result<()> {
        let Some(token) = self.tokenizer.next() else {
            return Err(Error::UnexpectedToken);
        };

        let is_open = token.is_open_brace();
        let is_close = token.is_close_brace();

        if !is_open && !is_close {
            return Err(Error::UnexpectedToken);
        }

        Ok(())
    }

    fn read_token(&mut self) -> Result<Token<'a>> {
        match self.tokenizer.next() {
            Some(token) => {
                if !token.is_valid() {
                    return Err(Error::InvalidToken);
                }

                Ok(token)
            },
            None => Err(Error::NoToken),
        }
    }

    fn read_float(&mut self) -> Result<Float> {
        let token = self.read_token()?;
        let parsed = token.parse::<Float>()?;
        Ok(parsed)
    }

    fn read_point(&mut self) -> Result<[Float; 3]> {
        let x = self.read_float()?;
        let y = self.read_float()?;
        let z = self.read_float()?;

        Ok([x, y, z])
    }

    fn read_matrix(&mut self) -> Result<[Float; 16]> {
        let mut m = [0.0; 16];
        for m in &mut m {
            *m = self.read_float()?;
        }
        Ok(m)
    }

    fn read_str(&mut self) -> Result<&'a str> {
        let token = self.read_token()?;
        token.unquote().ok_or(Error::InvalidString)
    }

    fn read_param(&mut self) -> Result<Param<'a>> {
        let type_and_name = self.read_str()?;

        let mut start = self.tokenizer.offset();
        let end;

        let value = self.read_token()?;

        if value.is_open_brace() {
            start = self.tokenizer.offset();

            loop {
                let value = self.read_token()?;

                if value.is_close_brace() {
                    end = self.tokenizer.offset() - 1;
                    break;
                }

                if value.is_directive() {
                    return Err(Error::UnexpectedToken);
                }
            }
        } else {
            end = start + value.token_size() + 1;
        }

        let token = self.tokenizer.token(start, end);
        let param = Param::new(type_and_name, token.value())?;

        Ok(param)
    }

    #[inline]
    fn read_param_list(&mut self) -> Result<ParamList<'a>> {
        let mut list = ParamList::default();

        loop {
            match self.tokenizer.peek_token() {
                Some(token) if token.is_quote() => {
                    let param = self.read_param()?;
                    list.add(param)?;
                },
                Some(_) => break,
                None => break,
            }
        }

        Ok(list)
    }
}
