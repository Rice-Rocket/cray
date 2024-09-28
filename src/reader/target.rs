use std::{collections::HashMap, fmt::Display, sync::{Arc, Mutex}};

use arrayvec::ArrayVec;
use string_interner::{DefaultBackend, StringInterner};

use crate::{color::{rgb_xyz::ColorEncodingCache, spectrum::Spectrum}, mipmap::MIPMap, options::Options, texture::TexInfo, Float};

use super::{error::{ParseError, ParseResult}, param::ParamList, paramdict::ParsedParameter};

pub type ParsedParameterVector = ArrayVec<ParsedParameter, 16>;

impl<'a> ParamList<'a> {
    pub fn parse(self) -> ParseResult<ParsedParameterVector> {
        let mut params = ArrayVec::new();

        for param in self.0.into_iter() {
            let param: ParsedParameter = param.1.parse()?;
            params.push(param);
        }

        Ok(params)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct FileLoc {
    pub(crate) filename: String,
    pub(crate) offset: usize,
    pub(crate) line: i32,
    pub(crate) n_lines: i32,
    /// The starting column of the span
    pub(crate) start: i32,
    /// The length of the span
    pub(crate) len: i32,
}

impl Default for FileLoc {
    fn default() -> Self {
        Self {
            filename: String::default(),
            offset: 0,
            line: 0,
            n_lines: 1,
            start: 0,
            len: 1,
        }
    }
}

impl FileLoc {
    pub fn new<S: ToString>(filename: S, offset: usize, line: i32, n_lines: i32, start: i32, len: i32) -> FileLoc {
        FileLoc {
            filename: filename.to_string(),
            offset,
            line,
            n_lines,
            start,
            len,
        }
    }

    pub fn from_file<S: ToString>(filename: S) -> FileLoc {
        FileLoc {
            filename: filename.to_string(),
            offset: 0,
            line: 0,
            n_lines: 0,
            start: 0,
            len: 0,
        }
    }
}

impl Display for FileLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.filename, self.line, self.start)
    }
}

pub trait ParserTarget {
    fn shape(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn option(&mut self, name: &str, value: &str, options: &mut Options, loc: FileLoc) -> ParseResult<()>;
    fn identity(&mut self, loc: FileLoc) -> ParseResult<()>;
    fn translate(&mut self, dx: Float, dy: Float, dz: Float, loc: FileLoc) -> ParseResult<()>;
    fn scale(&mut self, sx: Float, sy: Float, sz: Float, loc: FileLoc) -> ParseResult<()>;
    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float, loc: FileLoc) -> ParseResult<()>;
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
    ) -> ParseResult<()>;
    fn transform(&mut self, transform: [Float; 16], loc: FileLoc) -> ParseResult<()>;
    fn concat_transform(&mut self, transform: [Float; 16], loc: FileLoc) -> ParseResult<()>;
    fn coordinate_system(&mut self, name: &str, loc: FileLoc);
    fn coordinate_sys_transform(&mut self, name: &str, loc: FileLoc);
    fn active_transform_all(&mut self, loc: FileLoc);
    fn active_transform_end_time(&mut self, loc: FileLoc);
    fn active_transform_start_time(&mut self, loc: FileLoc);
    fn transform_times(&mut self, start: Float, end: Float, loc: FileLoc);
    fn color_space(&mut self, n: &str, loc: FileLoc) -> ParseResult<()>;
    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn camera(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    );
    fn make_named_medium(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: FileLoc,
    ) -> ParseResult<()>;
    fn medium_interface(&mut self, inside_name: &str, outside_name: &str, loc: FileLoc);
    fn sampler(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
    );
    fn world_begin(
        &mut self,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    );
    fn attribute_begin(&mut self, loc: FileLoc);
    fn attribute_end(&mut self, loc: FileLoc) -> ParseResult<()>;
    fn attribute(&mut self, target: &str, params: ParsedParameterVector, loc: FileLoc) -> ParseResult<()>;
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
    ) -> ParseResult<()>;
    fn material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    );
    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        options: &Options,
    ) -> ParseResult<()>;
    fn named_material(&mut self, name: &str, loc: FileLoc);
    fn light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner<DefaultBackend>,
        loc: FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    );
    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn reverse_orientation(&mut self, loc: FileLoc);
    fn object_begin(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> ParseResult<()>;
    fn object_end(&mut self, loc: FileLoc) -> ParseResult<()>;
    fn object_instance(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner<DefaultBackend>) -> ParseResult<()>;
    fn end_of_files(&mut self) -> ParseResult<()>;
}
