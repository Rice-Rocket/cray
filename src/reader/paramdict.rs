use std::{collections::HashMap, fmt::Display, str::FromStr as _, sync::Arc};

use crate::{color::{colorspace::{NamedColorSpace, RgbColorSpace}, named_spectrum::NamedSpectrum, rgb_xyz::Rgb, spectrum::{BlackbodySpectrum, PiecewiseLinearSpectrum, RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, Spectrum}}, error, texture::{FloatConstantTexture, FloatTexture, SpectrumConstantTexture, SpectrumTexture}, warn, Float, Normal3f, Point2f, Point3f, Vec2f, Vec3f};

use super::{error::{ParseError, ParseResult}, param::{Param, ParamType}, target::{FileLoc, ParsedParameterVector}, utils::{dequote_string, is_quoted_string}};

pub trait ParameterType: sealed::Sealed {
    const TYPE_NAME: &'static str;
    const N_PER_ITEM: i32;
    type ConvertType;
    type ReturnType;

    fn convert(v: &[Self::ConvertType], loc: &FileLoc) -> Self::ReturnType;
    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType>;
}

struct BooleanParam;
struct FloatParam;
struct IntegerParam;
struct Point2fParam;
struct Vec2fParam;
struct Point3fParam;
struct Vec3fParam;
struct Normal3fParam;
struct RgbParam;
struct SpectrumParam;
struct StringParam;
struct TextureParam;

impl ParameterType for BooleanParam {
    const TYPE_NAME: &'static str = "bool";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = bool;
    type ReturnType = bool;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.bools
    }
}

impl ParameterType for FloatParam {
    const TYPE_NAME: &'static str = "float";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = Float;
    type ReturnType = Float;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for IntegerParam {
    const TYPE_NAME: &'static str = "integer";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = i32;
    type ReturnType = i32;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.ints
    }
}

impl ParameterType for Point2fParam {
    const TYPE_NAME: &'static str = "point2";

    const N_PER_ITEM: i32 = 2;

    type ConvertType = Float;
    type ReturnType = Point2f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Point2f::new(v[0], v[1])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Vec2fParam {
    const TYPE_NAME: &'static str = "vector2";

    const N_PER_ITEM: i32 = 2;

    type ConvertType = Float;
    type ReturnType = Vec2f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Vec2f::new(v[0], v[1])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Point3fParam {
    const TYPE_NAME: &'static str = "point3";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Point3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Point3f::new(v[0], v[1], v[2])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Vec3fParam {
    const TYPE_NAME: &'static str = "vector3";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Vec3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Vec3f::new(v[0], v[1], v[2])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Normal3fParam {
    const TYPE_NAME: &'static str = "normal";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Normal3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Normal3f::new(v[0], v[1], v[2])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for RgbParam {
    const TYPE_NAME: &'static str = "rgb";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Rgb;

    fn convert(v: &[Self::ConvertType], loc: &FileLoc) -> Self::ReturnType {
        Rgb::new(v[0], v[1], v[2])
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for StringParam {
    const TYPE_NAME: &'static str = "string";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = String;
    type ReturnType = String;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0].clone()
    }

    fn get_values(param: &ParsedParameter) -> &Vec<Self::ConvertType> {
        &param.strings
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
    Unknown,
    Scalar,
    Integer,
    String,
    Boolean,
}

#[derive(Debug, Clone)]
pub struct ParsedParameter {
    pub name: String,
    pub param_type: String,
    pub loc: FileLoc,
    pub floats: Vec<Float>,
    pub ints: Vec<i32>,
    pub strings: Vec<String>,
    pub bools: Vec<bool>,
    pub looked_up: bool,
    pub color_space: Option<Arc<RgbColorSpace>>,
    pub may_be_unused: bool,
}

impl ParsedParameter {
    pub fn add_float(&mut self, v: Float) {
        assert!(self.ints.is_empty() && self.strings.is_empty() && self.bools.is_empty());
        self.floats.push(v);
    }

    pub fn add_int(&mut self, v: i32) {
        assert!(self.floats.is_empty() && self.strings.is_empty() && self.bools.is_empty());
        self.ints.push(v);
    }

    pub fn add_string(&mut self, v: String) {
        assert!(self.floats.is_empty() && self.ints.is_empty() && self.bools.is_empty());
        self.strings.push(v);
    }

    pub fn add_bool(&mut self, v: bool) {
        assert!(self.floats.is_empty() && self.ints.is_empty() && self.strings.is_empty());
        self.bools.push(v);
    }
}

impl<'a> Param<'a> {
    pub fn parse(self) -> ParseResult<ParsedParameter> {
        let name = self.name;
        let param_type = match self.ty {
            ParamType::Boolean => BooleanParam::TYPE_NAME.to_owned(),
            ParamType::Float => FloatParam::TYPE_NAME.to_owned(),
            ParamType::Integer => IntegerParam::TYPE_NAME.to_owned(),
            ParamType::Point2 => Point2fParam::TYPE_NAME.to_owned(),
            ParamType::Point3 => Point3fParam::TYPE_NAME.to_owned(),
            ParamType::Vec2 => Vec2fParam::TYPE_NAME.to_owned(),
            ParamType::Vec3 => Vec3fParam::TYPE_NAME.to_owned(),
            ParamType::Normal3 => Normal3fParam::TYPE_NAME.to_owned(),
            ParamType::Spectrum => "spectrum".to_owned(),
            ParamType::Rgb => "rgb".to_owned(),
            ParamType::Blackbody => "blackbody".to_owned(),
            ParamType::String => StringParam::TYPE_NAME.to_owned(),
            ParamType::Texture => "texture".to_owned(),
        };

        let mut param = ParsedParameter {
            name: name.to_owned(),
            param_type,
            loc: self.loc.clone(),
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            bools: Vec::new(),
            looked_up: false,
            color_space: None,
            may_be_unused: false,
        };

        let mut val_type = if param.param_type.as_str() == "integer" {
            ValueType::Integer
        } else {
            ValueType::Unknown
        };

        let mut split_values = Vec::new();
        let mut quoted_sequence = Vec::new();

        for v in self.value.split_ascii_whitespace() {
            if v.starts_with('\"') && v.ends_with('\"') {
                split_values.push(v.to_owned());
            } else if v.starts_with('\"') {
                quoted_sequence.push(v);
            } else if v.ends_with('\"') {
                quoted_sequence.push(v);
                let merged_quoted_string = quoted_sequence.join(" ");
                split_values.push(merged_quoted_string);
                quoted_sequence.clear();
            } else if !quoted_sequence.is_empty() {
                quoted_sequence.push(v);
            } else {
                split_values.push(v.to_owned());
            }
        }

        for v in split_values.into_iter() {
            if is_quoted_string(v.as_str()) {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::String,
                    ValueType::String => (),
                    _ => { error!(&self.loc, InvalidParameter, "parameter '{}' has mixed types", name); },
                }
                param.add_string(dequote_string(v.as_str()).to_owned());
            } else if v.starts_with('t') && v == "true" {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Boolean,
                    ValueType::Boolean => {}
                    _ => { error!(&self.loc, InvalidParameter, "parameter '{}' has mixed types", name); },
                }
                param.add_bool(true);
            } else if v.starts_with('f') && v == "false" {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Boolean,
                    ValueType::Boolean => {}
                    _ => { error!(&self.loc, InvalidParameter, "parameter '{}' has mixed types", name); },
                }
                param.add_bool(false);
            } else {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Scalar,
                    ValueType::Scalar => {}
                    ValueType::Integer => {}
                    _ => { error!(&self.loc, InvalidParameter, "parameter '{}' has mixed types", name); },
                }

                if val_type == ValueType::Integer {
                    param.add_int(v.parse::<i32>().expect("Expected integer"));
                } else {
                    param.add_float(v.parse::<Float>().expect("Expected float"));
                }
            }
        }

        Ok(param)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
}

impl Display for SpectrumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectrumType::Illuminant => write!(f, "Illuminant"),
            SpectrumType::Albedo => write!(f, "Albedo"),
            SpectrumType::Unbounded => write!(f, "Unbounded"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParameterDictionary {
    pub params: ParsedParameterVector,
    pub color_space: Arc<RgbColorSpace>,
    pub n_owned_params: i32,
}

impl Default for ParameterDictionary {
    fn default() -> Self {
        Self {
            params: ParsedParameterVector::default(),
            color_space: RgbColorSpace::get_named(NamedColorSpace::SRgb).clone(),
            n_owned_params: 0,
        }
    }
}

impl ParameterDictionary {
    pub fn new(
        params: ParsedParameterVector,
        color_space: Arc<RgbColorSpace>,
    ) -> ParameterDictionary {
        let n_owned_params = params.len() as i32;
        let d = ParameterDictionary {
            params: params.into_iter().rev().collect(),
            color_space,
            n_owned_params,
        };

        d.check_parameter_types();
        d
    }

    pub fn new_with_unowned(
        mut params: ParsedParameterVector,
        params_2: ParsedParameterVector,
        color_space: Arc<RgbColorSpace>
    ) -> ParameterDictionary {
        let n_owned_params = params.len() as i32;
        params = params.into_iter().rev().collect();
        params.extend(params_2.into_iter().rev());
        let d = ParameterDictionary {
            params,
            color_space,
            n_owned_params,
        };

        d.check_parameter_types();
        d
    }

    pub fn check_parameter_types(&self) -> ParseResult<()> {
        for p in &self.params {
            match p.param_type.as_str() {
                BooleanParam::TYPE_NAME => {
                    if p.bools.is_empty() {
                        error!(&p.loc, MissingValue, "no boolean values provided for boolean-valued parameter!");
                    }
                }
                FloatParam::TYPE_NAME
                | IntegerParam::TYPE_NAME
                | Point2fParam::TYPE_NAME
                | Vec2fParam::TYPE_NAME
                | Point3fParam::TYPE_NAME
                | Vec3fParam::TYPE_NAME
                | Normal3fParam::TYPE_NAME
                | "rgb"
                | "blackbody" => {
                    if p.ints.is_empty() && p.floats.is_empty() {
                        error!(
                            &p.loc,
                            InvalidValue,
                            "non-numeric values provided for numeric-valued parameter!",
                        );
                    }
                }
                StringParam::TYPE_NAME | "texture" => {
                    if p.strings.is_empty() {
                        error!(
                            &p.loc,
                            InvalidValue,
                            "non-string values provided for string-valued parameter!",
                        );
                    }
                }
                "spectrum" => {
                    if p.strings.is_empty() && p.ints.is_empty() && p.floats.is_empty() {
                        error!(&p.loc, InvalidValue, "expecting string or numeric-valued parameter for spectrum parameter.");
                    }
                }
                param => {
                    error!(&p.loc, UnknownValue, "unknown parameter type '{}'", param);
                }
            }
        }

        Ok(())
    }

    fn lookup_single<P: ParameterType>(
        &mut self,
        name: &str,
        default_value: P::ReturnType,
    ) -> ParseResult<P::ReturnType> {
        for p in &mut self.params {
            if p.name != name || p.param_type != P::TYPE_NAME {
                continue;
            }

            p.looked_up = true;
            let values = P::get_values(p);

            if values.is_empty() {
                error!(&p.loc, MissingValue, "no values provided for parameter '{}'", p.name);
            }

            if values.len() != P::N_PER_ITEM as usize {
                error!(
                    &p.loc,
                    InvalidValueCount,
                    "expected {} values for parameter '{}'",
                    P::N_PER_ITEM,
                    p.name
                );
            }

            return Ok(P::convert(values.as_slice(), &p.loc));
        }

        Ok(default_value)
    }

    pub fn get_one_float(&mut self, name: &str, default_value: Float) -> ParseResult<Float> {
        self.lookup_single::<FloatParam>(name, default_value)
    }

    pub fn get_one_int(&mut self, name: &str, default_value: i32) -> ParseResult<i32> {
        self.lookup_single::<IntegerParam>(name, default_value)
    }

    pub fn get_one_bool(&mut self, name: &str, default_value: bool) -> ParseResult<bool> {
        self.lookup_single::<BooleanParam>(name, default_value)
    }

    pub fn get_one_point2f(&mut self, name: &str, default_value: Point2f) -> ParseResult<Point2f> {
        self.lookup_single::<Point2fParam>(name, default_value)
    }

    pub fn get_one_vector2f(&mut self, name: &str, default_value: Vec2f) -> ParseResult<Vec2f> {
        self.lookup_single::<Vec2fParam>(name, default_value)
    }

    pub fn get_one_point3f(&mut self, name: &str, default_value: Point3f) -> ParseResult<Point3f> {
        self.lookup_single::<Point3fParam>(name, default_value)
    }

    pub fn get_one_vector3f(&mut self, name: &str, default_value: Vec3f) -> ParseResult<Vec3f> {
        self.lookup_single::<Vec3fParam>(name, default_value)
    }

    pub fn get_one_normal3f(&mut self, name: &str, default_value: Normal3f) -> ParseResult<Normal3f> {
        self.lookup_single::<Normal3fParam>(name, default_value)
    }

    pub fn get_one_rgb(&mut self, name: &str, default_value: Rgb) -> ParseResult<Rgb> {
        self.lookup_single::<RgbParam>(name, default_value)
    }

    pub fn get_one_string(&mut self, name: &str, default_value: &str) -> ParseResult<String> {
        self.lookup_single::<StringParam>(name, default_value.to_owned())
    }

    pub fn get_one_spectrum(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<Option<Arc<Spectrum>>> {
        let p = self.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            let s = Self::extract_spectrum_array(
                p,
                spectrum_type,
                self.color_space.clone(),
                cached_spectra,
            )?;
            if !s.is_empty() {
                if s.len() > 1 {
                    error!(
                        &p.loc,
                        InvalidValueCount,
                        "more than one value provided for parameter '{}'",
                        p.name
                    );
                }
                return Ok(Some(s.into_iter().nth(0).expect("Expected non-empty vector")));
            }

            Ok(default_value)
        } else {
            Ok(default_value)
        }
    }

    fn extract_spectrum_array(
        param: &mut ParsedParameter,
        spectrum_type: SpectrumType,
        color_space: Arc<RgbColorSpace>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>
    ) -> ParseResult<Vec<Arc<Spectrum>>> {
        if param.param_type == "rgb" {
            // TODO We could also handle "color" in this block with an upgrade option, but
            //  I don't intend to use old PBRT scene files for now.

            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                3,
                |v: &[Float], loc: &FileLoc| -> ParseResult<Arc<Spectrum>> {
                    let rgb = Rgb::new(v[0], v[1], v[2]);
                    let cs = if let Some(cs) = &param.color_space {
                        cs.clone()
                    } else {
                        color_space.clone()
                    };
                    if rgb.r < 0.0 || rgb.g < 0.0 || rgb.b < 0.0 {
                        error!(
                            loc,
                            InvalidValue,
                            "rgb parameter '{}' has negative component",
                            param.name
                        );
                    }
                    Ok(match spectrum_type {
                        SpectrumType::Illuminant => {
                            Arc::new(Spectrum::RgbIlluminant(
                                RgbIlluminantSpectrum::new(&cs, &rgb),
                            ))
                        }
                        SpectrumType::Albedo => {
                            if rgb.r > 1.0 || rgb.g > 1.0 || rgb.b > 1.0 {
                                error!(
                                    loc,
                                    InvalidValue,
                                    "rgb parameter '{}' has component value > 1.0",
                                    param.name
                                );
                            }
                            Arc::new(Spectrum::RgbAlbedo(RgbAlbedoSpectrum::new(
                                &cs, &rgb,
                            )))
                        }
                        SpectrumType::Unbounded => Arc::new(Spectrum::RgbUnbounded(
                            RgbUnboundedSpectrum::new(&cs, &rgb),
                        )),
                    })
                },
            );
        } else if param.param_type == "blackbody" {
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |v: &[Float], _loc: &FileLoc| -> ParseResult<Arc<Spectrum>> {
                    Ok(Arc::new(Spectrum::Blackbody(BlackbodySpectrum::new(v[0]))))
                },
            );
        } else if param.param_type == "spectrum" && !param.floats.is_empty() {
            if param.floats.len() % 2 != 0 {
                error!(
                    param.loc,
                    InvalidValueCount,
                    "found odd number of values for '{}'",
                    param.name
                );
            }
            let n_samples = param.floats.len() / 2;
            if n_samples == 1 {
                warn!(param.loc, "{} specified spectrum is only non-zero at a single wavelength; probably unintended", param.name);
            }
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                param.floats.len() as i32,
                |v: &[Float], _loc: &FileLoc| -> ParseResult<Arc<Spectrum>> {
                    let mut lambda = vec![0.0; n_samples];
                    let mut value = vec![0.0; n_samples];
                    for i in 0..n_samples {
                        if i > 0 && v[2 * i] <= lambda[i - 1] {
                            error!(param.loc, InvalidValue, "spectrum description invalid: at {}'th entry, wavelengths aren't increasing: {} >= {}", i - 1, lambda[i -1], v[2 * i]);
                        }
                        lambda[i] = v[2 * i];
                        value[i] = v[2 * i + 1];
                    }
                    return Ok(Arc::new(Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::new(
                                    lambda.as_slice(),
                                    value.as_slice(),
                    ))));
                },
                );
        } else if param.param_type == "spectrum" && !param.strings.is_empty() {
            return Self::return_array(
                param.strings.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |s: &[String], loc: &FileLoc| -> ParseResult<Arc<Spectrum>> {
                    let named_spectrum = NamedSpectrum::from_str(&s[0]);
                    if let Ok(named_spectrum) = named_spectrum {
                        return Ok(Spectrum::get_named_spectrum(named_spectrum));
                    }

                    let Some(spd) = Spectrum::read(&s[0], cached_spectra) else {
                        error!(@file &s[0], InvalidFileContents, "unable to read/invalid spectrum file");
                    };

                    Ok(spd)
                },
            );
        } else {
            return Ok(Vec::new());
        }
    }

    pub fn get_spectrum_array(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<Vec<Arc<Spectrum>>> {
        let p = self.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            let s = Self::extract_spectrum_array(
                p,
                spectrum_type,
                self.color_space.clone(),
                cached_spectra,
            )?;
            if !s.is_empty() {
                return Ok(s);
            }
        }
        Ok(Vec::new())
    }

    fn return_array<ValueType, ReturnType, C>(
        values: &[ValueType],
        loc: &FileLoc,
        name: &str,
        looked_up: &mut bool,
        n_per_item: i32,
        mut convert: C,
    ) -> ParseResult<Vec<ReturnType>>
    where
        C: FnMut(&[ValueType], &FileLoc) -> ParseResult<ReturnType,>
    {
        if values.is_empty() {
            error!(loc, MissingValue, "no values provided for '{}'", name);
        }
        if values.len() % n_per_item as usize != 0 {
            error!(
                loc,
                InvalidValueCount,
                "number of values provided for '{}' is not a multiple of {}",
                name, n_per_item
            );
        }

        *looked_up = true;
        let n = values.len() / n_per_item as usize;

        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(convert(&values[n_per_item as usize * i..], loc)?);
        }
        Ok(v)
    }

    fn lookup_array<P: ParameterType>(&mut self, name: &str) -> ParseResult<Vec<P::ReturnType>> {
        for p in &mut self.params {
            if p.name == name && p.param_type == P::TYPE_NAME {
                let mut looked_up = p.looked_up;
                let to_return = Self::return_array(
                    P::get_values(p),
                    &p.loc,
                    &p.name,
                    &mut looked_up,
                    P::N_PER_ITEM,
                    |v, loc| {
                        Ok(P::convert(v, loc))
                    },
                )?;
                p.looked_up = looked_up;
                return Ok(to_return);
            }
        }
        Ok(Vec::new())
    }

    pub fn get_float_array(&mut self, name: &str) -> ParseResult<Vec<Float>> {
        self.lookup_array::<FloatParam>(name)
    }

    pub fn get_int_array(&mut self, name: &str) -> ParseResult<Vec<i32>> {
        self.lookup_array::<IntegerParam>(name)
    }

    pub fn get_bool_array(&mut self, name: &str) -> ParseResult<Vec<bool>> {
        self.lookup_array::<BooleanParam>(name)
    }

    pub fn get_point2f_array(&mut self, name: &str) -> ParseResult<Vec<Point2f>> {
        self.lookup_array::<Point2fParam>(name)
    }

    pub fn get_vector2f_array(&mut self, name: &str) -> ParseResult<Vec<Vec2f>> {
        self.lookup_array::<Vec2fParam>(name)
    }

    pub fn get_point3f_array(&mut self, name: &str) -> ParseResult<Vec<Point3f>> {
        self.lookup_array::<Point3fParam>(name)
    }

    pub fn get_vector3f_array(&mut self, name: &str) -> ParseResult<Vec<Vec3f>> {
        self.lookup_array::<Vec3fParam>(name)
    }

    pub fn get_normal3f_array(&mut self, name: &str) -> ParseResult<Vec<Normal3f>> {
        self.lookup_array::<Normal3fParam>(name)
    }

    pub fn get_rgb_array(&mut self, name: &str) -> ParseResult<Vec<Rgb>> {
        self.lookup_array::<RgbParam>(name)
    }
}

#[derive(Debug, Clone, Default)]
pub struct NamedTextures {
    pub float_textures: HashMap<String, Arc<FloatTexture>>,
    pub albedo_spectrum_textures: HashMap<String, Arc<SpectrumTexture>>,
    pub unbounded_spectrum_textures: HashMap<String, Arc<SpectrumTexture>>,
    pub illuminant_spectrum_textures: HashMap<String, Arc<SpectrumTexture>>,
}

pub struct TextureParameterDictionary {
    pub dict: ParameterDictionary,
}

impl TextureParameterDictionary {
    pub fn new(dict: ParameterDictionary) -> Self {
        Self { dict }
    }

    pub fn get_one_float(&mut self, name: &str, default_value: Float) -> ParseResult<Float> {
        self.dict.get_one_float(name, default_value)
    }

    pub fn get_one_int(&mut self, name: &str, default_value: i32) -> ParseResult<i32> {
        self.dict.get_one_int(name, default_value)
    }

    pub fn get_one_bool(&mut self, name: &str, default_value: bool) -> ParseResult<bool> {
        self.dict.get_one_bool(name, default_value)
    }

    pub fn get_one_point2f(&mut self, name: &str, default_value: Point2f) -> ParseResult<Point2f> {
        self.dict.get_one_point2f(name, default_value)
    }

    pub fn get_one_vector2f(&mut self, name: &str, default_value: Vec2f) -> ParseResult<Vec2f> {
        self.dict.get_one_vector2f(name, default_value)
    }

    pub fn get_one_point3f(&mut self, name: &str, default_value: Point3f) -> ParseResult<Point3f> {
        self.dict.get_one_point3f(name, default_value)
    }

    pub fn get_one_vector3f(&mut self, name: &str, default_value: Vec3f) -> ParseResult<Vec3f> {
        self.dict.get_one_vector3f(name, default_value)
    }

    pub fn get_one_normal3f(&mut self, name: &str, default_value: Normal3f) -> ParseResult<Normal3f> {
        self.dict.get_one_normal3f(name, default_value)
    }

    pub fn get_one_rgb(&mut self, name: &str, default_value: Rgb) -> ParseResult<Rgb> {
        self.dict.get_one_rgb(name, default_value)
    }

    pub fn get_one_spectrum(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<Option<Arc<Spectrum>>> {
        self.dict.get_one_spectrum(name, default_value, spectrum_type, cached_spectra)
    }

    pub fn get_one_string(&mut self, name: &str, default_value: &str) -> ParseResult<String> {
        self.dict.get_one_string(name, default_value)
    }

    pub fn get_float_array(&mut self, name: &str) -> ParseResult<Vec<Float>> {
        self.dict.get_float_array(name)
    }

    pub fn get_int_array(&mut self, name: &str) -> ParseResult<Vec<i32>> {
        self.dict.get_int_array(name)
    }

    pub fn get_bool_array(&mut self, name: &str) -> ParseResult<Vec<bool>> {
        self.dict.get_bool_array(name)
    }

    pub fn get_point2f_array(&mut self, name: &str) -> ParseResult<Vec<Point2f>> {
        self.dict.get_point2f_array(name)
    }

    pub fn get_vector2f_array(&mut self, name: &str) -> ParseResult<Vec<Vec2f>> {
        self.dict.get_vector2f_array(name)
    }

    pub fn get_point3f_array(&mut self, name: &str) -> ParseResult<Vec<Point3f>> {
        self.dict.get_point3f_array(name)
    }

    pub fn get_vector3f_array(&mut self, name: &str) -> ParseResult<Vec<Vec3f>> {
        self.dict.get_vector3f_array(name)
    }

    pub fn get_normal3f_array(&mut self, name: &str) -> ParseResult<Vec<Normal3f>> {
        self.dict.get_normal3f_array(name)
    }

    pub fn get_rgb_array(&mut self, name: &str) -> ParseResult<Vec<Rgb>> {
        self.dict.get_rgb_array(name)
    }

    pub fn get_spectrum_array(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> ParseResult<Vec<Arc<Spectrum>>> {
        self.dict
            .get_spectrum_array(name, spectrum_type, cached_spectra)
    }

    pub fn get_string_array(&mut self, name: &str) -> ParseResult<Vec<String>> {
        self.dict.lookup_array::<StringParam>(name)
    }

    pub fn get_float_texture(
        &mut self,
        name: &str,
        default_value: Float,
        textures: &NamedTextures,
    ) -> ParseResult<Arc<FloatTexture>> {
        let tex = self.get_float_texture_or_none(name, textures)?;
        if let Some(tex) = tex {
            Ok(tex)
        } else {
            Ok(Arc::new(FloatTexture::Constant(FloatConstantTexture::new(
                default_value,
            ))))
        }
    }

    pub fn get_float_texture_or_none(
        &mut self,
        name: &str,
        textures: &NamedTextures,
    ) -> ParseResult<Option<Arc<FloatTexture>>> {
        let Some(p) = self.dict.params.iter_mut().find(|p| p.name == name) else {
            return Ok(None);
        };
        if p.param_type == "texture" {
            if p.strings.is_empty() {
                error!(
                    p.loc,
                    MissingValue,
                    "no filename provided for texture parameter '{}'",
                    p.name
                );
            }
            if p.strings.len() != 1 {
                error!(
                    p.loc,
                    ValueConflict,
                    "more than one filename provided for texture parameter '{}'",
                    p.name
                );
            }

            p.looked_up = true;

            let tex = textures.float_textures.get(p.strings[0].as_str());
            if let Some(tex) = tex {
                Ok(Some(tex.clone()))
            } else {
                error!(p.loc, UndefinedValue, "couldn't find float texture '{}'", p.strings[0]);
            }
        } else if p.param_type == "float" {
            let v = self.get_one_float(name, 0.0)?;
            return Ok(Some(Arc::new(FloatTexture::Constant(FloatConstantTexture::new(
                v,
            )))));
        } else {
            Ok(None)
        }
    }

    pub fn get_spectrum_texture(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ParseResult<Option<Arc<SpectrumTexture>>> {
        let tex = self.get_spectrum_texture_or_none(name, spectrum_type, cached_spectra, textures)?;
        if let Some(tex) = tex {
            Ok(Some(tex))
        } else if let Some(default_value) = default_value {
            return Ok(Some(Arc::new(SpectrumTexture::Constant(
                SpectrumConstantTexture::new(default_value.clone()),
            ))));
        } else {
            return Ok(None);
        }
    }

    pub fn get_spectrum_texture_or_none(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ParseResult<Option<Arc<SpectrumTexture>>> {
        let spectrum_textures = match spectrum_type {
            SpectrumType::Illuminant => &textures.illuminant_spectrum_textures,
            SpectrumType::Albedo => &textures.albedo_spectrum_textures,
            SpectrumType::Unbounded => &textures.unbounded_spectrum_textures,
        };

        let p = self.dict.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            match p.param_type.as_str() {
                "texture" => {
                    if p.strings.is_empty() {
                        error!(
                            p.loc,
                            MissingValue,
                            "no filename provided for texture parameter '{}'",
                            p.name
                        );
                    }
                    if p.strings.len() > 1 {
                        error!(
                            p.loc,
                            ValueConflict,
                            "more than one filename provided for texture parameter '{}'",
                            p.name
                        );
                    }

                    p.looked_up = true;

                    let spec = spectrum_textures.get(p.strings[0].as_str());
                    if let Some(spec) = spec {
                        Ok(Some(spec.clone()))
                    } else {
                        error!(p.loc, UndefinedValue, "couldn't find spectrum texture '{}'", p.strings[0]);
                    }
                }
                "rgb" => {
                    if p.floats.len() != 3 {
                        error!(
                            p.loc,
                            InvalidValueCount,
                            "expected 3 values for rgb texture parameter '{}'",
                            p.name
                        );
                    }
                    p.looked_up = true;
                    let rgb = Rgb::new(p.floats[0], p.floats[1], p.floats[2]);
                    if rgb.r < 0.0 || rgb.g < 0.0 || rgb.b < 0.0 {
                        error!(p.loc, InvalidValue, "rgb parameter '{}' has negative component", p.name);
                    }
                    let s = match spectrum_type {
                        SpectrumType::Illuminant => Spectrum::RgbIlluminant(
                            RgbIlluminantSpectrum::new(&self.dict.color_space, &rgb),
                        ),
                        SpectrumType::Albedo => Spectrum::RgbAlbedo(
                            RgbAlbedoSpectrum::new(&self.dict.color_space, &rgb),
                        ),
                        SpectrumType::Unbounded => Spectrum::RgbUnbounded(
                            RgbUnboundedSpectrum::new(&self.dict.color_space, &rgb),
                        ),
                    };
                    Ok(Some(Arc::new(SpectrumTexture::Constant(
                        SpectrumConstantTexture::new(Arc::new(s)),
                    ))))
                }
                "spectrum" | "blackbody" => {
                    let loc = p.loc.clone();
                    let Some(s) = self.get_one_spectrum(name, None, spectrum_type, cached_spectra)?.clone() else {
                        error!(loc, MissingValue, "expected spectrum '{}'", name);
                    };

                    Ok(Some(Arc::new(SpectrumTexture::Constant(
                        SpectrumConstantTexture::new(s),
                    ))))
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::BooleanParam {}
    impl Sealed for super::FloatParam {}
    impl Sealed for super::IntegerParam {}
    impl Sealed for super::Point2fParam {}
    impl Sealed for super::Vec2fParam {}
    impl Sealed for super::Point3fParam {}
    impl Sealed for super::Vec3fParam {}
    impl Sealed for super::Normal3fParam {}
    impl Sealed for super::RgbParam {}
    impl Sealed for super::SpectrumParam {}
    impl Sealed for super::StringParam {}
    impl Sealed for super::TextureParam {}
}
