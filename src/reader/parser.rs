use std::{collections::HashMap, env, fs::{self, File}, io::{BufReader, Read}, path::Path, slice, sync::{Arc, Mutex}};

use flate2::bufread::GzDecoder;
use string_interner::{DefaultBackend, StringInterner};

use crate::{clear_log, color::{rgb_xyz::ColorEncodingCache, spectrum::Spectrum}, error, file::set_search_directory, log, mipmap::MIPMap, new_syntax_err, options::Options, texture::TexInfo, Float};

use super::{error::{ParseResult, SyntaxError, SyntaxErrorKind}, param::{Param, ParamList}, target::{FileLoc, ParserTarget}, token::{Directive, Token}, tokenizer::Tokenizer};

pub fn parse_files<T: ParserTarget>(
    files: &[&str],
    target: &mut T,
    options: &mut Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
) -> ParseResult<()> {
    if files.is_empty() {
        todo!("read from stdin when no files are provided")
    }

    for file in files {
        if *file != "-" {
            set_search_directory(options, file);
        }

        let data = fs::read_to_string(file).unwrap();
        parse_str(&data, target, options, string_interner, cached_spectra, texture_cache, gamma_encoding_cache, file.to_string())?;
    }

    Ok(())
}

pub fn parse_str<T: ParserTarget>(
    data: &str,
    target: &mut T,
    options: &mut Options,
    string_interner: &mut StringInterner<DefaultBackend>,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
    gamma_encoding_cache: &mut ColorEncodingCache,
    filename: String,
) -> ParseResult<()> {
    let mut parsers = Vec::new();
    parsers.push(Parser::new(data, filename));

    let mut includes = Vec::new();

    while let Some(parser) = parsers.last_mut() {
        let element = match parser.parse_next() {
            Ok(element) => element,
            Err(err) => {
                if matches!(err.kind, SyntaxErrorKind::EndOfFile) {
                    clear_log!();
                    parsers.pop();
                    continue;
                } else {
                    println!("{}", err.format(parser.tokenizer.str));
                    error!(@noloc "");
                }
            }
        };

        match element {
            Element::Include { file: path_name, loc } => {
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
                    }, path_name.to_string())
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
                    }, path_name.to_string())
                };

                log!("Parsing included scene file '{}'", path_name);

                parsers.push(parser);
            },
            Element::Import { .. } => todo!("Support import directive"),
            Element::Option { param, loc } => target.option(param.name, param.value, options, loc)?,
            Element::Film { ty, params, loc } => target.film(ty, params.parse()?, string_interner, loc),
            Element::ColorSpace { ty, loc } => target.color_space(ty, loc)?,
            Element::Camera { ty, params, loc } => target.camera(ty, params.parse()?, string_interner, loc, options),
            Element::Sampler { ty, params, loc } => target.sampler(ty, params.parse()?, string_interner, loc),
            Element::Integrator { ty, params, loc } => target.integrator(ty, params.parse()?, string_interner, loc),
            Element::Accelerator { ty, params, loc } => target.accelerator(ty, params.parse()?, string_interner, loc),
            Element::CoordinateSystem { name, loc } => target.coordinate_system(name, loc),
            Element::CoordSysTransform { name, loc } => target.coordinate_sys_transform(name, loc),
            Element::PixelFilter { name, params, loc } => target.pixel_filter(name, params.parse()?, string_interner, loc),
            Element::Identity { loc } => target.identity(loc)?,
            Element::Translate { v, loc } => target.translate(v[0], v[1], v[2], loc)?,
            Element::Scale { v, loc } => target.scale(v[0], v[1], v[2], loc)?,
            Element::Rotate { angle, v, loc } => target.rotate(angle, v[0], v[1], v[2], loc)?,
            Element::LookAt { eye, look_at, up, loc } => target.look_at(
                eye[0], eye[1], eye[2], look_at[0], look_at[1], look_at[2],
                up[0], up[1], up[2], loc,
            )?,
            Element::Transform { m, loc } => target.transform(m, loc)?,
            Element::ConcatTransform { m, loc } => target.concat_transform(m, loc)?,
            Element::TransformTimes { start, end, loc } => target.transform_times(start, end, loc),
            Element::ActiveTransform { ty, loc } => match ty {
                "StartTime" => target.active_transform_start_time(loc),
                "EndTime" => target.active_transform_end_time(loc),
                "All" => target.active_transform_all(loc),
                _ => todo!("Unknown active transform type"),
            },
            Element::ReverseOrientation { loc } => target.reverse_orientation(loc),
            Element::WorldBegin { loc } => target.world_begin(string_interner, loc, options),
            Element::AttributeBegin { loc } => target.attribute_begin(loc),
            Element::AttributeEnd { loc } => target.attribute_end(loc)?,
            Element::Attribute { attr_target, params, loc } => target.attribute(attr_target, params.parse()?, loc)?,
            Element::LightSource { ty, params, loc } => target.light_source(
                ty, params.parse()?, string_interner, loc,
                cached_spectra, options,
            ),
            Element::AreaLightSource { ty, params, loc } => target.area_light_source(ty, params.parse()?, loc),
            Element::Material { ty, params, loc } => target.material(ty, params.parse()?, string_interner, loc, options),
            Element::MakeNamedMaterial { name, params, loc } => target.make_named_material(name, params.parse()?, string_interner, loc, options)?,
            Element::NamedMaterial { name, loc } => target.named_material(name, loc),
            Element::Texture { name, ty, class, params, loc } => target.texture(
                name, ty, class, params.parse()?, string_interner, loc,
                options, cached_spectra, texture_cache, gamma_encoding_cache,
            )?,
            Element::Shape { name, params, loc } => target.shape(name, params.parse()?, string_interner, loc),
            Element::ObjectBegin { name, loc } => target.object_begin(name, loc, string_interner)?,
            Element::ObjectEnd { loc } => target.object_end(loc)?,
            Element::ObjectInstance { name, loc } => target.object_instance(name, loc, string_interner)?,
            Element::MakeNamedMedium { name, params, loc } => target.make_named_medium(name, params.parse()?, string_interner, cached_spectra, loc)?,
            Element::MediumInterface { interior, exterior, loc } => target.medium_interface(interior, exterior, loc),
        }
    }

    target.end_of_files()
}

#[derive(Debug, PartialEq)]
pub enum Element<'a> {
    /// Include behaves similarly to the #include directive in C++: parsing of the current file is suspended,
    /// the specified file is parsed in its entirety, and only then does parsing of the current file resume.
    /// Its effect is equivalent to direct text substitution of the included file.
    Include {
        file: &'a str,
        loc: FileLoc,
    },
    Import {
        file: &'a str,
        loc: FileLoc,
    },
    Option {
        param: Param<'a>,
        loc: FileLoc,
    },
    Film {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    ColorSpace {
        ty: &'a str,
        loc: FileLoc,
    },
    Camera {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    Sampler {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    Integrator {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    Accelerator {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    CoordinateSystem {
        name: &'a str,
        loc: FileLoc,
    },
    CoordSysTransform {
        name: &'a str,
        loc: FileLoc,
    },
    PixelFilter {
        name: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    Identity {
        loc: FileLoc,
    },
    /// `Translate x y z`
    Translate {
        v: [Float; 3],
        loc: FileLoc,
    },
    /// `Scale x y z`
    Scale {
        v: [Float; 3],
        loc: FileLoc,
    },
    /// `Rotate angle x y z`
    Rotate {
        angle: Float,
        v: [Float; 3],
        loc: FileLoc,
    },
    /// `LookAt eye_x eye_y eye_z look_x look_y look_z up_x up_y up_z`
    LookAt {
        eye: [Float; 3],
        look_at: [Float; 3],
        up: [Float; 3],
        loc: FileLoc,
    },
    /// `Transform m00 ... m33`
    Transform {
        m: [Float; 16],
        loc: FileLoc,
    },
    /// `ConcatTransform m00 .. m33`
    ConcatTransform {
        m: [Float; 16],
        loc: FileLoc,
    },
    /// `TransformTimes start end`.
    TransformTimes {
        start: Float,
        end: Float,
        loc: FileLoc,
    },
    ActiveTransform {
        ty: &'a str,
        loc: FileLoc,
    },
    /// `ReverseOrientation`.
    ReverseOrientation {
        loc: FileLoc,
    },
    /// `WorldBegin`
    WorldBegin {
        loc: FileLoc,
    },
    /// `AttributeBegin`
    AttributeBegin {
        loc: FileLoc,
    },
    /// `AttributeEnd`
    AttributeEnd {
        loc: FileLoc,
    },
    /// `Attribute "target" parameter-list`
    Attribute {
        attr_target: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    LightSource {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    AreaLightSource {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    Material {
        ty: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    MakeNamedMaterial {
        name: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    NamedMaterial {
        name: &'a str,
        loc: FileLoc,
    },
    /// `Texture "name" "type" "class" [ parameter-list ]`
    Texture {
        name: &'a str,
        ty: &'a str,
        class: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    /// `Shape "name" parameter-list`
    Shape {
        name: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    ObjectBegin {
        name: &'a str,
        loc: FileLoc,
    },
    ObjectEnd {
        loc: FileLoc,
    },
    ObjectInstance {
        name: &'a str,
        loc: FileLoc,
    },
    MakeNamedMedium {
        name: &'a str,
        params: ParamList<'a>,
        loc: FileLoc,
    },
    MediumInterface {
        interior: &'a str,
        exterior: &'a str,
        loc: FileLoc,
    },
}

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(str: &'a str, filename: String) -> Self {
        let tokenizer = Tokenizer::new(str, filename);
        Self { tokenizer }
    }

    pub fn parse_next(&mut self) -> Result<Element<'a>, SyntaxError> {
        let Some(next_token) = self.tokenizer.next() else {
            return Err(new_syntax_err!(EndOfFile, self.tokenizer.loc()));
        };

        let loc = next_token.loc();

        let directive = next_token.directive()
            .ok_or(new_syntax_err!(SyntaxErrorKind::UnknownDirective(next_token.value().to_owned()), loc.clone()))?;

        let element = match directive {
            Directive::Include => Element::Include { file: self.read_str()?, loc },
            Directive::Import => Element::Import { file: self.read_str()?, loc },
            Directive::Option => Element::Option { param: self.read_param()?, loc },
            Directive::Film => Element::Film { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::ColorSpace => Element::ColorSpace { ty: self.read_str()?, loc },
            Directive::Camera => Element::Camera { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::Sampler => Element::Sampler { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::Integrator => Element::Integrator { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::Accelerator => Element::Accelerator { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::CoordinateSystem => Element::CoordinateSystem { name: self.read_str()?, loc },
            Directive::CoordSysTransform => Element::CoordSysTransform { name: self.read_str()?, loc },
            Directive::PixelFilter => Element::PixelFilter { name: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::Identity => Element::Identity { loc },
            Directive::Translate => Element::Translate { v: self.read_point()?, loc },
            Directive::Scale => Element::Scale { v: self.read_point()?, loc },
            Directive::Rotate => Element::Rotate { angle: self.read_float()?, v: self.read_point()?, loc },
            Directive::LookAt => Element::LookAt { eye: self.read_point()?, look_at: self.read_point()?, up: self.read_point()?, loc },
            Directive::Transform => {
                self.skip_brace()?;
                let elem = Element::Transform { m: self.read_matrix()?, loc };
                self.skip_brace()?;
                elem
            },
            Directive::ConcatTransform => {
                self.skip_brace()?;
                let elem = Element::ConcatTransform { m: self.read_matrix()?, loc };
                self.skip_brace()?;
                elem
            },
            Directive::TransformTimes => Element::TransformTimes { start: self.read_float()?, end: self.read_float()?, loc },
            Directive::ActiveTransform => Element::ActiveTransform { ty: self.read_str()?, loc },
            Directive::ReverseOrientation => Element::ReverseOrientation { loc },
            Directive::WorldBegin => Element::WorldBegin { loc },
            Directive::AttributeBegin => Element::AttributeBegin { loc },
            Directive::AttributeEnd => Element::AttributeEnd { loc },
            Directive::Attribute => Element::Attribute { attr_target: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::LightSource => Element::LightSource { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::AreaLightSource => Element::AreaLightSource { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::Material => Element::Material { ty: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::MakeNamedMaterial => Element::MakeNamedMaterial { name: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::NamedMaterial => Element::NamedMaterial { name: self.read_str()?, loc },
            Directive::Texture => Element::Texture {
                name: self.read_str()?,
                ty: self.read_str()?,
                class: self.read_str()?,
                params: self.read_param_list()?,
                loc,
            },
            Directive::Shape => Element::Shape { name: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::ObjectBegin => Element::ObjectBegin { name: self.read_str()?, loc },
            Directive::ObjectEnd => Element::ObjectEnd { loc },
            Directive::ObjectInstance => Element::ObjectInstance { name: self.read_str()?, loc },
            Directive::MakeNamedMedium => Element::MakeNamedMedium { name: self.read_str()?, params: self.read_param_list()?, loc },
            Directive::MediumInterface => Element::MediumInterface { interior: self.read_str()?, exterior: self.read_str()?, loc },
        };

        Ok(element)
    }

    fn skip_brace(&mut self) -> Result<(), SyntaxError> {
        let Some(token) = self.tokenizer.next() else {
            return Err(new_syntax_err!(UnexpectedToken, self.tokenizer.loc()));
        };

        let is_open = token.is_open_brace();
        let is_close = token.is_close_brace();

        if !is_open && !is_close {
            return Err(new_syntax_err!(UnexpectedToken, self.tokenizer.loc(), "expected '[' or ']'"));
        }

        Ok(())
    }

    fn read_token(&mut self) -> Result<Token<'a>, SyntaxError> {
        match self.tokenizer.next() {
            Some(token) => {
                if !token.is_valid() {
                    return Err(new_syntax_err!(InvalidToken, token.loc(), "{}", token.value()));
                }

                Ok(token)
            },
            None => Err(new_syntax_err!(NoToken, self.tokenizer.loc())),
        }
    }

    fn read_float(&mut self) -> Result<Float, SyntaxError> {
        let token = self.read_token()?;
        let parsed = token.parse::<Float>()?;
        Ok(parsed)
    }

    fn read_point(&mut self) -> Result<[Float; 3], SyntaxError> {
        let x = self.read_float()?;
        let y = self.read_float()?;
        let z = self.read_float()?;

        Ok([x, y, z])
    }

    fn read_matrix(&mut self) -> Result<[Float; 16], SyntaxError> {
        let mut m = [0.0; 16];
        for m in &mut m {
            *m = self.read_float()?;
        }
        Ok(m)
    }

    fn read_str(&mut self) -> Result<&'a str, SyntaxError> {
        let token = self.read_token()?;
        token.unquote().ok_or(new_syntax_err!(InvalidString, token.loc()))
    }

    fn read_param(&mut self) -> Result<Param<'a>, SyntaxError> {
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
                    return Err(new_syntax_err!(UnexpectedToken, value.loc(), "{}", value.value()));
                }
            }
        } else {
            end = start + value.token_size() + 1;
        }

        let token = self.tokenizer.token(start, end);
        let param = Param::new(type_and_name, token.value(), token.loc())?;

        Ok(param)
    }

    #[inline]
    fn read_param_list(&mut self) -> Result<ParamList<'a>, SyntaxError> {
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
