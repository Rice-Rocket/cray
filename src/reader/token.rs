use std::fmt;
use std::str::FromStr;

use crate::new_syntax_err;

use super::{error::{SyntaxError, SyntaxErrorKind}, target::FileLoc};

#[derive(Debug, PartialEq, Eq)]
pub struct Token<'a> {
    str: &'a str,
    loc: FileLoc,
}

impl<'a> Token<'a> {
    pub fn new(str: &'a str, loc: FileLoc) -> Self {
        Token { str, loc }
    }

    pub fn token_size(&self) -> usize {
        self.str.len()
    }

    pub fn value(&self) -> &'a str {
        self.str.trim()
    }

    pub fn loc(&self) -> FileLoc {
        self.loc.clone()
    }

    pub fn parse<F: FromStr>(&self) -> Result<F, <F as FromStr>::Err> {
        self.str.parse::<F>()
    }

    pub fn is_quote(&self) -> bool {
        self.str.len() >= 2 && self.str.starts_with('\"') && self.str.ends_with('\"')
    }

    pub fn is_directive(&self) -> bool {
        Directive::from_str(self.str).is_ok()
    }

    pub fn directive(&self) -> Option<Directive> {
        Directive::from_str(self.str).ok()
    }

    pub fn is_open_brace(&self) -> bool {
        self.str == "["
    }

    pub fn is_close_brace(&self) -> bool {
        self.str == "]"
    }

    pub fn unquote(&self) -> Option<&'a str> {
        if self.is_quote() {
            let len = self.str.len();
            Some(&self.str[1..len - 1])
        } else {
            None
        }
    }

    pub fn is_valid(&self) -> bool {
        if self.str.is_empty() {
            return false;
        }

        let starts_with_quote = self.str.starts_with('\"');
        let ends_with_quote = self.str.ends_with('\"');

        if starts_with_quote || ends_with_quote {
            if starts_with_quote != ends_with_quote {
                return false;
            }

            if self.str.len() < 2 {
                return false;
            }
        }

        if !starts_with_quote && self.str.contains(' ') {
            return false;
        }

        true
    }
}

#[derive(Debug, PartialEq)]
pub enum Directive {
    Identity,
    Translate,
    Scale,
    Rotate,
    LookAt,
    CoordinateSystem,
    CoordSysTransform,
    Transform,
    ConcatTransform,
    TransformTimes,
    ActiveTransform,

    Include,
    Import,

    Option,

    Camera,
    Sampler,
    ColorSpace,
    Film,
    Integrator,
    Accelerator,
    PixelFilter,

    MakeNamedMedium,
    MediumInterface,

    WorldBegin,

    AttributeBegin,
    AttributeEnd,
    Attribute,

    Shape,
    ReverseOrientation,
    ObjectBegin,
    ObjectEnd,
    ObjectInstance,

    LightSource,
    AreaLightSource,

    Material,
    Texture,
    MakeNamedMaterial,
    NamedMaterial,
}

impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for Directive {
    type Err = SyntaxError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let e = match s {
            "Identity" => Directive::Identity,
            "Translate" => Directive::Translate,
            "Scale" => Directive::Scale,
            "Rotate" => Directive::Rotate,
            "LookAt" => Directive::LookAt,
            "CoordinateSystem" => Directive::CoordinateSystem,
            "CoordSysTransform" => Directive::CoordSysTransform,
            "Transform" => Directive::Transform,
            "ConcatTransform" => Directive::ConcatTransform,
            "TransformTimes" => Directive::TransformTimes,
            "ActiveTransform" => Directive::ActiveTransform,
            "Include" => Directive::Include,
            "Import" => Directive::Import,
            "Option" => Directive::Option,
            "Camera" => Directive::Camera,
            "Sampler" => Directive::Sampler,
            "ColorSpace" => Directive::ColorSpace,
            "Film" => Directive::Film,
            "Integrator" => Directive::Integrator,
            "Accelerator" => Directive::Accelerator,
            "MakeNamedMedium" => Directive::MakeNamedMedium,
            "MediumInterface" => Directive::MediumInterface,
            "WorldBegin" => Directive::WorldBegin,
            "AttributeBegin" => Directive::AttributeBegin,
            "AttributeEnd" => Directive::AttributeEnd,
            "Attribute" => Directive::Attribute,
            "Shape" => Directive::Shape,
            "ReverseOrientation" => Directive::ReverseOrientation,
            "ObjectBegin" => Directive::ObjectBegin,
            "ObjectEnd" => Directive::ObjectEnd,
            "ObjectInstance" => Directive::ObjectInstance,
            "LightSource" => Directive::LightSource,
            "AreaLightSource" => Directive::AreaLightSource,
            "Material" => Directive::Material,
            "Texture" => Directive::Texture,
            "MakeNamedMaterial" => Directive::MakeNamedMaterial,
            "NamedMaterial" => Directive::NamedMaterial,
            "PixelFilter" => Directive::PixelFilter,
            _ => return Err(new_syntax_err!(SyntaxErrorKind::UnknownDirective(s.to_owned()), FileLoc::default()))
        };

        Ok(e)
    }
}
