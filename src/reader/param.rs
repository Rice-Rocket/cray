use std::{collections::HashMap, num::{ParseFloatError, ParseIntError}, str::{FromStr, ParseBoolError}};

use super::error::{Error, Result};

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

impl FromStr for ParamType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
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
            _ => return Err(Error::InvalidParamType),
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
}

impl<'a> Param<'a> {
    pub fn new(type_and_name: &'a str, value: &'a str) -> Result<Self> {
        let mut split = type_and_name.split_whitespace();

        let ty_name = split.next().ok_or(Error::InvalidParamName)?;
        let ty = ParamType::from_str(ty_name)?;

        let name = split.next().ok_or(Error::InvalidParamName)?;

        Ok(Self { name, ty, value })
    }

    pub fn items<T: FromStr>(&self) -> impl Iterator<Item = std::result::Result<T, <T as FromStr>::Err>> + 'a {
        self.value.split_whitespace().map(|s| T::from_str(s))
    }

    pub fn rgb(&self) -> Result<[f32; 3]> {
        let mut iter = self.items::<f32>();

        let r = iter.next().ok_or(Error::MissingRequiredParameter)??;
        let g = iter.next().ok_or(Error::MissingRequiredParameter)??;
        let b = iter.next().ok_or(Error::MissingRequiredParameter)??;

        Ok([r, g, b])
    }

    pub fn single<T: FromStr>(&self) -> std::result::Result<T, <T as FromStr>::Err> {
        T::from_str(self.value)
    }

    pub fn vec<T: FromStr>(&self) -> std::result::Result<Vec<T>, <T as FromStr>::Err> {
        self.items().collect::<std::result::Result<Vec<T>, <T as FromStr>::Err>>()
    }

    pub fn spectrum(&self) -> Result<Spectrum> {
        let res = match self.ty {
            ParamType::Rgb => Spectrum::Rgb(self.rgb()?),
            ParamType::Blackbody => Spectrum::Blackbody(self.single()?),
            _ => return Err(Error::InvalidObjectType),
        };

        Ok(res)
    }
}

#[derive(Default, Debug, PartialEq, Clone)]
pub struct ParamList<'a>(pub HashMap<&'a str, Param<'a>>);

impl<'a> ParamList<'a> {
    pub fn add(&mut self, param: Param<'a>) -> Result<()> {
        if self.0.insert(param.name, param).is_some() {
            return Err(Error::DuplicatedParamName);
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
