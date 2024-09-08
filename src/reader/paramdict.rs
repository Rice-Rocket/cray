use std::{collections::HashMap, fmt::Display, str::FromStr as _, sync::Arc};

use tracing::warn;

use crate::{color::{colorspace::{NamedColorSpace, RgbColorSpace}, named_spectrum::NamedSpectrum, rgb_xyz::Rgb, spectrum::{BlackbodySpectrum, PiecewiseLinearSpectrum, RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, Spectrum}}, texture::{FloatConstantTexture, FloatTexture, SpectrumConstantTexture, SpectrumTexture}, Normal3f, Point2f, Point3f, Float, Vec2f, Vec3f};

use super::{param::{Param, ParamType}, target::{FileLoc, ParsedParameterVector}, utils::{dequote_string, is_quoted_string}};

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

impl<'a> From<Param<'a>> for ParsedParameter {
    fn from(value: Param<'a>) -> Self {
        let name = value.name;
        let param_type = match value.ty {
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

        let loc = FileLoc::default();

        let mut param = ParsedParameter {
            name: name.to_owned(),
            param_type,
            loc,
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

        for v in value.value.split_ascii_whitespace() {
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
                    _ => panic!("parameter {} has mixed types", name),
                }
                param.add_string(dequote_string(v.as_str()).to_owned());
            } else if v.starts_with('t') && v == "true" {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Boolean,
                    ValueType::Boolean => {}
                    _ => panic!("Parameter {} has mixed types", name),
                }
                param.add_bool(true);
            } else if v.starts_with('f') && v == "false" {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Boolean,
                    ValueType::Boolean => {}
                    _ => panic!("Parameter {} has mixed types", name),
                }
                param.add_bool(false);
            } else {
                match val_type {
                    ValueType::Unknown => val_type = ValueType::Scalar,
                    ValueType::Scalar => {}
                    ValueType::Integer => {}
                    _ => panic!("Parameter {} has mixed types", name),
                }

                if val_type == ValueType::Integer {
                    param.add_int(v.parse::<i32>().expect("Expected integer"));
                } else {
                    param.add_float(v.parse::<Float>().expect("Expected float"));
                }
            }
        }

        param
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

    pub fn check_parameter_types(&self) {
        for p in &self.params {
            match p.param_type.as_str() {
                BooleanParam::TYPE_NAME => {
                    if p.bools.is_empty() {
                        panic!("No boolean values provided for boolean-valued parameter!");
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
                        panic!(
                            "{} Non-numeric values provided for numeric-valued parameter!",
                            p.loc
                        );
                    }
                }
                StringParam::TYPE_NAME | "texture" => {
                    if p.strings.is_empty() {
                        panic!(
                            "{} Non-string values provided for string-valued parameter!",
                            p.loc
                        )
                    }
                }
                "spectrum" => {
                    if p.strings.is_empty() && p.ints.is_empty() && p.floats.is_empty() {
                        panic!("{} Expecting string or numeric-valued parameter for spectrum parameter.", p.loc)
                    }
                }
                param => {
                    panic!("{} Unknown parameter type {}", p.loc, param)
                }
            }
        }
    }

    fn lookup_single<P: ParameterType>(
        &mut self,
        name: &str,
        default_value: P::ReturnType,
    ) -> P::ReturnType {
        for p in &mut self.params {
            if p.name != name || p.param_type != P::TYPE_NAME {
                continue;
            }

            p.looked_up = true;
            let values = P::get_values(p);

            if values.is_empty() {
                panic!("{} no values provided for parameter {}", p.loc, p.name);
            }

            if values.len() != P::N_PER_ITEM as usize {
                panic!(
                    "{} expected {} values for parameter {}",
                    p.loc,
                    P::N_PER_ITEM,
                    p.name
                );
            }

            return P::convert(values.as_slice(), &p.loc);
        }

        default_value
    }

    pub fn get_one_float(&mut self, name: &str, default_value: Float) -> Float {
        self.lookup_single::<FloatParam>(name, default_value)
    }

    pub fn get_one_int(&mut self, name: &str, default_value: i32) -> i32 {
        self.lookup_single::<IntegerParam>(name, default_value)
    }

    pub fn get_one_bool(&mut self, name: &str, default_value: bool) -> bool {
        self.lookup_single::<BooleanParam>(name, default_value)
    }

    pub fn get_one_point2f(&mut self, name: &str, default_value: Point2f) -> Point2f {
        self.lookup_single::<Point2fParam>(name, default_value)
    }

    pub fn get_one_vector2f(&mut self, name: &str, default_value: Vec2f) -> Vec2f {
        self.lookup_single::<Vec2fParam>(name, default_value)
    }

    pub fn get_one_point3f(&mut self, name: &str, default_value: Point3f) -> Point3f {
        self.lookup_single::<Point3fParam>(name, default_value)
    }

    pub fn get_one_vector3f(&mut self, name: &str, default_value: Vec3f) -> Vec3f {
        self.lookup_single::<Vec3fParam>(name, default_value)
    }

    pub fn get_one_normal3f(&mut self, name: &str, default_value: Normal3f) -> Normal3f {
        self.lookup_single::<Normal3fParam>(name, default_value)
    }

    pub fn get_one_rgb(&mut self, name: &str, default_value: Rgb) -> Rgb {
        self.lookup_single::<RgbParam>(name, default_value)
    }

    pub fn get_one_string(&mut self, name: &str, default_value: &str) -> String {
        self.lookup_single::<StringParam>(name, default_value.to_owned())
    }

    pub fn get_one_spectrum(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Option<Arc<Spectrum>> {
        let p = self.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            let s = Self::extract_spectrum_array(
                p,
                spectrum_type,
                self.color_space.clone(),
                cached_spectra,
            );
            if !s.is_empty() {
                if s.len() > 1 {
                    panic!(
                        "{} More than one value provided for parameter {}",
                        p.loc, p.name
                    );
                }
                return Some(s.into_iter().nth(0).expect("Expected non-empty vector"));
            }

            default_value
        } else {
            default_value
        }
    }

    fn extract_spectrum_array(
        param: &mut ParsedParameter,
        spectrum_type: SpectrumType,
        color_space: Arc<RgbColorSpace>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>
    ) -> Vec<Arc<Spectrum>> {
        if param.param_type == "rgb" {
            // TODO We could also handle "color" in this block with an upgrade option, but
            //  I don't intend to use old PBRT scene files for now.

            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                3,
                |v: &[Float], loc: &FileLoc| -> Arc<Spectrum> {
                    let rgb = Rgb::new(v[0], v[1], v[2]);
                    let cs = if let Some(cs) = &param.color_space {
                        cs.clone()
                    } else {
                        color_space.clone()
                    };
                    if rgb.r < 0.0 || rgb.g < 0.0 || rgb.b < 0.0 {
                        panic!(
                            "{} Rgb Parameter {} has negative component",
                            loc, param.name
                        );
                    }
                    match spectrum_type {
                        SpectrumType::Illuminant => {
                            Arc::new(Spectrum::RgbIlluminant(
                                RgbIlluminantSpectrum::new(&cs, &rgb),
                            ))
                        }
                        SpectrumType::Albedo => {
                            if rgb.r > 1.0 || rgb.g > 1.0 || rgb.b > 1.0 {
                                panic!(
                                    "{} Rgb Parameter {} has component value > 1.0",
                                    loc, param.name
                                );
                            }
                            Arc::new(Spectrum::RgbAlbedo(RgbAlbedoSpectrum::new(
                                &cs, &rgb,
                            )))
                        }
                        SpectrumType::Unbounded => Arc::new(Spectrum::RgbUnbounded(
                            RgbUnboundedSpectrum::new(&cs, &rgb),
                        )),
                    }
                },
            );
        } else if param.param_type == "blackbody" {
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |v: &[Float], _loc: &FileLoc| -> Arc<Spectrum> {
                    Arc::new(Spectrum::Blackbody(BlackbodySpectrum::new(v[0])))
                },
            );
        } else if param.param_type == "spectrum" && !param.floats.is_empty() {
            if param.floats.len() % 2 != 0 {
                panic!(
                    "{} Found odd number of values for {}",
                    param.loc, param.name
                );
            }
            let n_samples = param.floats.len() / 2;
            if n_samples == 1 {
                warn!("{} {} Specified spectrum is only non-zero at a single wavelength; probably unintended", param.loc, param.name);
            }
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                param.floats.len() as i32,
                |v: &[Float], _loc: &FileLoc| -> Arc<Spectrum> {
                    let mut lambda = vec![0.0; n_samples];
                    let mut value = vec![0.0; n_samples];
                    for i in 0..n_samples {
                        if i > 0 && v[2 * i] <= lambda[i - 1] {
                            panic!("{} Spectrum description invalid: at {}'th entry, wavelengths aren't increasing: {} >= {}", param.loc, i - 1, lambda[i -1], v[2 * i]);
                        }
                        lambda[i] = v[2 * i];
                        value[i] = v[2 * i + 1];
                    }
                    return Arc::new(Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::new(
                        lambda.as_slice(),
                        value.as_slice(),
                    )));
                },
            );
        } else if param.param_type == "spectrum" && !param.strings.is_empty() {
            return Self::return_array(
                param.strings.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |s: &[String], loc: &FileLoc| -> Arc<Spectrum> {
                    let named_spectrum = NamedSpectrum::from_str(&s[0]);
                    if let Ok(named_spectrum) = named_spectrum {
                        return Spectrum::get_named_spectrum(named_spectrum);
                    }

                    let spd = Spectrum::read(&s[0], cached_spectra);
                    if spd.is_none() {
                        panic!("{} Unable to read/invalid spectrum file {}", &s[0], loc);
                    }
                    spd.unwrap()
                },
            );
        } else {
            return Vec::new();
        }
    }

    pub fn get_spectrum_array(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Vec<Arc<Spectrum>> {
        let p = self.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            let s = Self::extract_spectrum_array(
                p,
                spectrum_type,
                self.color_space.clone(),
                cached_spectra,
            );
            if !s.is_empty() {
                return s;
            }
        }
        Vec::new()
    }

    fn return_array<ValueType, ReturnType, C>(
        values: &[ValueType],
        loc: &FileLoc,
        name: &str,
        looked_up: &mut bool,
        n_per_item: i32,
        mut convert: C,
    ) -> Vec<ReturnType>
    where
        C: FnMut(&[ValueType], &FileLoc) -> ReturnType,
    {
        if values.is_empty() {
            panic!("{} No values provided for {}", loc, name);
        }
        if values.len() % n_per_item as usize != 0 {
            panic!(
                "{} Number of values provided for {} is not a multiple of {}",
                loc, name, n_per_item
            );
        }

        *looked_up = true;
        let n = values.len() / n_per_item as usize;

        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(convert(&values[n_per_item as usize * i..], loc));
        }
        v
    }

    fn lookup_array<P: ParameterType>(&mut self, name: &str) -> Vec<P::ReturnType> {
        for p in &mut self.params {
            if p.name == name && p.param_type == P::TYPE_NAME {
                let mut looked_up = p.looked_up;
                let to_return = Self::return_array(
                    P::get_values(p),
                    &p.loc,
                    &p.name,
                    &mut looked_up,
                    P::N_PER_ITEM,
                    P::convert,
                );
                p.looked_up = looked_up;
                return to_return;
            }
        }
        Vec::new()
    }

    pub fn get_float_array(&mut self, name: &str) -> Vec<Float> {
        self.lookup_array::<FloatParam>(name)
    }

    pub fn get_int_array(&mut self, name: &str) -> Vec<i32> {
        self.lookup_array::<IntegerParam>(name)
    }

    pub fn get_bool_array(&mut self, name: &str) -> Vec<bool> {
        self.lookup_array::<BooleanParam>(name)
    }

    pub fn get_point2f_array(&mut self, name: &str) -> Vec<Point2f> {
        self.lookup_array::<Point2fParam>(name)
    }

    pub fn get_vector2f_array(&mut self, name: &str) -> Vec<Vec2f> {
        self.lookup_array::<Vec2fParam>(name)
    }

    pub fn get_point3f_array(&mut self, name: &str) -> Vec<Point3f> {
        self.lookup_array::<Point3fParam>(name)
    }

    pub fn get_vector3f_array(&mut self, name: &str) -> Vec<Vec3f> {
        self.lookup_array::<Vec3fParam>(name)
    }

    pub fn get_normal3f_array(&mut self, name: &str) -> Vec<Normal3f> {
        self.lookup_array::<Normal3fParam>(name)
    }

    pub fn get_rgb_array(&mut self, name: &str) -> Vec<Rgb> {
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

    pub fn get_one_float(&mut self, name: &str, default_value: Float) -> Float {
        self.dict.get_one_float(name, default_value)
    }

    pub fn get_one_int(&mut self, name: &str, default_value: i32) -> i32 {
        self.dict.get_one_int(name, default_value)
    }

    pub fn get_one_bool(&mut self, name: &str, default_value: bool) -> bool {
        self.dict.get_one_bool(name, default_value)
    }

    pub fn get_one_point2f(&mut self, name: &str, default_value: Point2f) -> Point2f {
        self.dict.get_one_point2f(name, default_value)
    }

    pub fn get_one_vector2f(&mut self, name: &str, default_value: Vec2f) -> Vec2f {
        self.dict.get_one_vector2f(name, default_value)
    }

    pub fn get_one_point3f(&mut self, name: &str, default_value: Point3f) -> Point3f {
        self.dict.get_one_point3f(name, default_value)
    }

    pub fn get_one_vector3f(&mut self, name: &str, default_value: Vec3f) -> Vec3f {
        self.dict.get_one_vector3f(name, default_value)
    }

    pub fn get_one_normal3f(&mut self, name: &str, default_value: Normal3f) -> Normal3f {
        self.dict.get_one_normal3f(name, default_value)
    }

    pub fn get_one_rgb(&mut self, name: &str, default_value: Rgb) -> Rgb {
        self.dict.get_one_rgb(name, default_value)
    }

    pub fn get_one_spectrum(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Option<Arc<Spectrum>> {
        self.dict
            .get_one_spectrum(name, default_value, spectrum_type, cached_spectra)
    }

    pub fn get_one_string(&mut self, name: &str, default_value: &str) -> String {
        self.dict.get_one_string(name, default_value)
    }

    pub fn get_float_array(&mut self, name: &str) -> Vec<Float> {
        self.dict.get_float_array(name)
    }

    pub fn get_int_array(&mut self, name: &str) -> Vec<i32> {
        self.dict.get_int_array(name)
    }

    pub fn get_bool_array(&mut self, name: &str) -> Vec<bool> {
        self.dict.get_bool_array(name)
    }

    pub fn get_point2f_array(&mut self, name: &str) -> Vec<Point2f> {
        self.dict.get_point2f_array(name)
    }

    pub fn get_vector2f_array(&mut self, name: &str) -> Vec<Vec2f> {
        self.dict.get_vector2f_array(name)
    }

    pub fn get_point3f_array(&mut self, name: &str) -> Vec<Point3f> {
        self.dict.get_point3f_array(name)
    }

    pub fn get_vector3f_array(&mut self, name: &str) -> Vec<Vec3f> {
        self.dict.get_vector3f_array(name)
    }

    pub fn get_normal3f_array(&mut self, name: &str) -> Vec<Normal3f> {
        self.dict.get_normal3f_array(name)
    }

    pub fn get_rgb_array(&mut self, name: &str) -> Vec<Rgb> {
        self.dict.get_rgb_array(name)
    }

    pub fn get_spectrum_array(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Vec<Arc<Spectrum>> {
        self.dict
            .get_spectrum_array(name, spectrum_type, cached_spectra)
    }

    pub fn get_string_array(&mut self, name: &str) -> Vec<String> {
        self.dict.lookup_array::<StringParam>(name)
    }

    pub fn get_float_texture(
        &mut self,
        name: &str,
        default_value: Float,
        textures: &NamedTextures,
    ) -> Arc<FloatTexture> {
        let tex = self.get_float_texture_or_none(name, textures);
        if let Some(tex) = tex {
            tex
        } else {
            Arc::new(FloatTexture::Constant(FloatConstantTexture::new(
                default_value,
            )))
        }
    }

    pub fn get_float_texture_or_none(
        &mut self,
        name: &str,
        textures: &NamedTextures,
    ) -> Option<Arc<FloatTexture>> {
        let p = self.dict.params.iter_mut().find(|p| p.name == name)?;
        if p.param_type == "texture" {
            if p.strings.is_empty() {
                panic!(
                    "{} No filename provided for texture parameter {}",
                    p.loc, p.name
                );
            }
            if p.strings.len() != 1 {
                panic!(
                    "{} More than one filename provided for texture parameter {}",
                    p.loc, p.name
                );
            }

            p.looked_up = true;

            let tex = textures.float_textures.get(p.strings[0].as_str());
            if let Some(tex) = tex {
                Some(tex.clone())
            } else {
                panic!("{} Couldn't find float texture {}", p.loc, p.strings[0]);
            }
        } else if p.param_type == "float" {
            let v = self.get_one_float(name, 0.0);
            return Some(Arc::new(FloatTexture::Constant(FloatConstantTexture::new(
                v,
            ))));
        } else {
            None
        }
    }

    pub fn get_spectrum_texture(
        &mut self,
        name: &str,
        default_value: Option<Arc<Spectrum>>,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> Option<Arc<SpectrumTexture>> {
        let tex = self.get_spectrum_texture_or_none(name, spectrum_type, cached_spectra, textures);
        if let Some(tex) = tex {
            Some(tex)
        } else if let Some(default_value) = default_value {
            return Some(Arc::new(SpectrumTexture::Constant(
                SpectrumConstantTexture::new(default_value.clone()),
            )));
        } else {
            return None;
        }
    }

    pub fn get_spectrum_texture_or_none(
        &mut self,
        name: &str,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> Option<Arc<SpectrumTexture>> {
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
                        panic!(
                            "{} No filename provided for texture parameter {}",
                            p.loc, p.name
                        );
                    }
                    if p.strings.len() > 1 {
                        panic!(
                            "{} More than one filename provided for texture parameter {}",
                            p.loc, p.name
                        );
                    }

                    p.looked_up = true;

                    let spec = spectrum_textures.get(p.strings[0].as_str());
                    if let Some(spec) = spec {
                        Some(spec.clone())
                    } else {
                        panic!("{} Couldn't find spectrum texture {}", p.loc, p.strings[0]);
                    }
                }
                "rgb" => {
                    if p.floats.len() != 3 {
                        panic!(
                            "{} Expected 3 values for Rgb texture parameter {}",
                            p.loc, p.name
                        );
                    }
                    p.looked_up = true;
                    let rgb = Rgb::new(p.floats[0], p.floats[1], p.floats[2]);
                    if rgb.r < 0.0 || rgb.g < 0.0 || rgb.b < 0.0 {
                        panic!("{} Rgb Parameter {} has negative component", p.loc, p.name);
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
                    Some(Arc::new(SpectrumTexture::Constant(
                        SpectrumConstantTexture::new(Arc::new(s)),
                    )))
                }
                "spectrum" | "blackbody" => {
                    let s = self.get_one_spectrum(name, None, spectrum_type, cached_spectra);
                    assert!(s.is_some());
                    let s = s.unwrap();
                    Some(Arc::new(SpectrumTexture::Constant(
                        SpectrumConstantTexture::new(s.clone()),
                    )))
                }
                _ => None,
            }
        } else {
            None
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
