use std::{collections::HashMap, num::{ParseFloatError, ParseIntError}, str::{FromStr, ParseBoolError}};

use crate::new_syntax_err;

use super::{error::{ParseOk, ParseResult, SyntaxError}, target::FileLoc};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParamType {
    Boolean,
    Float,
    Integer,
    Point2,
    Point3,
    Vec2,
    Vec3,
    Normal3,
    Spectrum,
    Rgb,
    Blackbody,
    String,
    Texture
}

impl std::fmt::Display for ParamType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ParamType::Boolean => "bool",
            ParamType::Float => "float",
            ParamType::Integer => "integer",
            ParamType::Point2 => "point2",
            ParamType::Point3 => "point3",
            ParamType::Vec2 => "vector2",
            ParamType::Vec3 => "vector3",
            ParamType::Normal3 => "normal",
            ParamType::Spectrum => "spectrum",
            ParamType::Rgb => "rgb",
            ParamType::Blackbody => "blackbody",
            ParamType::String => "string",
            ParamType::Texture => "texture",
        };

        write!(f, "{}", s)
    }
}

impl FromStr for ParamType {
    type Err = SyntaxError;

    fn from_str(s: &str) -> Result<ParamType, Self::Err> {
        let ty = match s {
            "bool" => ParamType::Boolean,
            "integer" => ParamType::Integer,
            "float" => ParamType::Float,
            "point2" => ParamType::Point2,
            "vector2" => ParamType::Vec2,
            "point3" => ParamType::Point3,
            "vector3" => ParamType::Vec3,
            "normal" => ParamType::Normal3,
            "normal3" => ParamType::Normal3,
            "spectrum" => ParamType::Spectrum,
            "rgb" => ParamType::Rgb,
            "blackbody" => ParamType::Blackbody,
            "string" => ParamType::String,
            "texture" => ParamType::Texture,
            s => return Err(new_syntax_err!(InvalidParamType, FileLoc::default(), "{s}")),
        };

        Ok(ty)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Spectrum {
    Rgb([f32; 3]),
    Blackbody(i32),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Param<'a> {
    pub name: &'a str,
    pub ty: ParamType,
    pub value: &'a str,
    pub loc: FileLoc,
}

impl<'a> Param<'a> {
    pub fn new(type_and_name: &'a str, value: &'a str, loc: FileLoc) -> Result<Self, SyntaxError> {
        let mut split = type_and_name.split_whitespace();

        let ty_name = split.next().ok_or(new_syntax_err!(InvalidParamName, loc.clone(), "{type_and_name}"))?;
        let ty = ParamType::from_str(ty_name)?;

        let name = split.next().ok_or(new_syntax_err!(InvalidParamName, loc.clone(), "{type_and_name}"))?;

        Ok(Self { name, ty, value, loc })
    }

    pub fn items<T: FromStr>(&self) -> impl Iterator<Item = std::result::Result<T, <T as FromStr>::Err>> + 'a {
        self.value.split_whitespace().map(|s| T::from_str(s))
    }

    pub fn rgb(&self) -> Result<[f32; 3], SyntaxError> {
        let mut iter = self.items::<f32>();

        let r = iter.next().ok_or(new_syntax_err!(MissingRequiredParameter, self.loc.clone()))??;
        let g = iter.next().ok_or(new_syntax_err!(MissingRequiredParameter, self.loc.clone()))??;
        let b = iter.next().ok_or(new_syntax_err!(MissingRequiredParameter, self.loc.clone()))??;

        Ok([r, g, b])
    }

    pub fn single<T: FromStr>(&self) -> std::result::Result<T, <T as FromStr>::Err> {
        T::from_str(self.value)
    }

    pub fn vec<T: FromStr>(&self) -> std::result::Result<Vec<T>, <T as FromStr>::Err> {
        self.items().collect::<std::result::Result<Vec<T>, <T as FromStr>::Err>>()
    }

    pub fn spectrum(&self) -> Result<Spectrum, SyntaxError> {
        let res = match self.ty {
            ParamType::Rgb => Spectrum::Rgb(self.rgb()?),
            ParamType::Blackbody => Spectrum::Blackbody(self.single()?),
            t => return Err(new_syntax_err!(InvalidObjectType, self.loc.clone(), "{t}"))
        };

        Ok(res)
    }
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct ParamList<'a>(pub HashMap<&'a str, Param<'a>>);

impl<'a> ParamList<'a> {
    pub fn add(&mut self, param: Param<'a>) -> Result<(), SyntaxError> {
        if self.0.insert(param.name, param.clone()).is_some() {
            return Err(new_syntax_err!(DuplicatedParamName, param.loc, "{}", param.name));
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Param<'a>> {
        self.0.get(name)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn vec<T: FromStr>(&self, name: &str) -> std::result::Result<Option<Vec<T>>, <T as FromStr>::Err> {
        let res = match self.get(name).map(|param| param.vec()) {
            Some(v) => Some(v?),
            None => None,
        };

        Ok(res)
    }

    pub fn floats(&self, name: &str) -> std::result::Result<Option<Vec<f32>>, ParseFloatError> {
        self.vec(name)
    }

    pub fn integers(&self, name: &str) -> std::result::Result<Option<Vec<i32>>, ParseIntError> {
        self.vec(name)
    }

    fn single<T: FromStr>(&self, name: &str, default: T) -> std::result::Result<T, <T as FromStr>::Err> {
        self.get(name)
            .map(|p| p.single::<T>())
            .unwrap_or(Ok(default))
    }

    /// Get a float value by name.
    ///
    /// If there is no parameter with name `name`, a `default` value will
    /// be returned.
    ///
    /// If there is a value and it's not possible to parse it into float,
    /// an error will be returned.
    pub fn float(&self, name: &str, default: f32) -> std::result::Result<f32, ParseFloatError> {
        self.single(name, default)
    }

    pub fn integer(&self, name: &str, default: i32) -> std::result::Result<i32, ParseIntError> {
        self.single(name, default)
    }

    pub fn boolean(&self, name: &str, default: bool) -> std::result::Result<bool, ParseBoolError> {
        self.single(name, default)
    }

    pub fn string(&self, name: &str) -> Option<&str> {
        self.get(name).map(|v| v.value)
    }

    pub fn extend(&mut self, other: &ParamList<'a>) {
        for (k, v) in &other.0 {
            self.0.insert(k, v.clone());
        }
    }
}
