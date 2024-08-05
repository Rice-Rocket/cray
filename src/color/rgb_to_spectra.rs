use once_cell::sync::Lazy;
use rgb2spec::RGB2Spec;

use crate::Float;

use super::rgb_xyz::Rgb;

#[derive(Debug, PartialEq, Clone)]
pub enum Gamut {
    SRgb,
    Xyz,
    ERgb,
    Aces2065_1,
    ProPhotoRgb,
    Rec2020,
}

pub fn get_rgb_to_spec(gamut: &Gamut, rgb: &Rgb) -> [Float; 3] {
    match gamut {
        Gamut::SRgb => Lazy::force(&SRGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::Xyz => Lazy::force(&XYZ_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::ERgb => Lazy::force(&ERGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::Aces2065_1 => Lazy::force(&ACES2065_1_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::ProPhotoRgb => Lazy::force(&PROPHOTORGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::Rec2020 => Lazy::force(&REC2020_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
    }
}

pub static SRGB_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/srgb.spec").unwrap());
pub static XYZ_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/xyz.spec").unwrap());
pub static ERGB_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/ergb.spec").unwrap());
pub static ACES2065_1_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/aces2065_1.spec").unwrap());
pub static PROPHOTORGB_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/prophotorgb.spec").unwrap());
pub static REC2020_RGB2SPEC: Lazy<RGB2Spec> = Lazy::new(|| RGB2Spec::load("rgbtospec/rec2020.spec").unwrap());
