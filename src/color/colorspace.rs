use std::{str::FromStr, sync::Arc};

use once_cell::sync::Lazy;

use crate::{color::rgb_to_spectra, error, mat::mul_mat_vec, reader::error::{ParseError, ParseResult}, Mat3, Point2f};

use super::{named_spectrum::NamedSpectrum, rgb_xyz::{Rgb, RgbSigmoidPolynomial, Xyz}, rgb_to_spectra::Gamut, spectrum::{DenselySampledSpectrum, Spectrum}};

#[derive(Debug, PartialEq, Clone)]
pub struct RgbColorSpace {
    pub r: Point2f,
    pub g: Point2f,
    pub b: Point2f,
    pub whitepoint: Point2f,
    pub illuminant: Arc<Spectrum>,
    pub xyz_from_rgb: Mat3,
    pub rgb_from_xyz: Mat3,
    gamut: Gamut,
}

impl RgbColorSpace {
    pub fn new(r: Point2f, g: Point2f, b: Point2f, illuminant: &Spectrum, gamut: Gamut) -> RgbColorSpace {
        let w = Xyz::from_spectrum(illuminant);

        let whitepoint = w.xy();
        let r_xyz = Xyz::from_xyy_default(&r);
        let g_xyz = Xyz::from_xyy_default(&g);
        let b_xyz = Xyz::from_xyy_default(&b);

        let rgb = Mat3::new(
            r_xyz.x, g_xyz.x, b_xyz.x,
            r_xyz.y, g_xyz.y, b_xyz.y,
            r_xyz.z, g_xyz.z, b_xyz.z,
        );

        let c = rgb.inverse() * w;

        let xyz_from_rgb = rgb * Mat3::new(
            c[0], 0.0, 0.0,
            0.0, c[1], 0.0,
            0.0, 0.0, c[2]
        );
        let rgb_from_xyz = xyz_from_rgb.inverse();

        let illuminant = DenselySampledSpectrum::new(illuminant);

        RgbColorSpace {
            r,
            g,
            b,
            whitepoint,
            illuminant: Arc::new(Spectrum::DenselySampled(illuminant)),
            xyz_from_rgb,
            rgb_from_xyz,
            gamut,
        }
    }

    pub fn get_named(cs: NamedColorSpace) -> &'static Arc<RgbColorSpace> {
        match cs {
            NamedColorSpace::SRgb => Lazy::force(&SRGB),
            NamedColorSpace::Rec2020 => Lazy::force(&REC_2020),
            NamedColorSpace::Aces2065_1 => Lazy::force(&ACES2065_1),
        }
    }

    pub fn to_rgb(&self, xyz: &Xyz) -> Rgb {
        mul_mat_vec::<3, Xyz, Rgb, Mat3>(&self.rgb_from_xyz, xyz)
    }

    pub fn to_xyz(&self, rgb: &Rgb) -> Xyz {
        mul_mat_vec::<3, Rgb, Xyz, Mat3>(&self.xyz_from_rgb, rgb)
    }

    pub fn to_rgb_coeffs(&self, rgb: &Rgb) -> RgbSigmoidPolynomial {
        debug_assert!(rgb.r >= 0.0 && rgb.g >= 0.0 && rgb.b >= 0.0);
        #[cfg(feature = "use_f64")]
        {
            RgbSigmoidPolynomial::from_array_f64(rgb_to_spectra::get_rgb_to_spec(&self.gamut, rgb))
        }
        #[cfg(not(feature = "use_f64"))]
        {
            RgbSigmoidPolynomial::from_array(rgb_to_spectra::get_rgb_to_spec(&self.gamut, rgb))
        }
    }

    pub fn convert_rgb_colorspace(&self, to: &RgbColorSpace) -> Mat3 {
        if self == to {
            return Mat3::IDENTITY;
        }

        to.rgb_from_xyz * self.xyz_from_rgb
    }

    pub fn luminance_vector(&self) -> Rgb {
        Rgb::new(
            self.xyz_from_rgb[(0, 0)],
            self.xyz_from_rgb[(0, 1)],
            self.xyz_from_rgb[(0, 2)]
        )
    }
}

pub enum NamedColorSpace {
    SRgb,
    Rec2020,
    Aces2065_1,
}

impl FromStr for NamedColorSpace {
    type Err = ParseError;

    fn from_str(value: &str) -> Result<NamedColorSpace, ParseError> {
        Ok(match value.to_ascii_lowercase().as_str() {
            "srgb" => NamedColorSpace::SRgb,
            "rec2020" => NamedColorSpace::Rec2020,
            "aces2065-1" => NamedColorSpace::Aces2065_1,
            _ => { error!(@noloc "unknown color space '{}'", value); },
        })
    }
}

pub static SRGB: Lazy<Arc<RgbColorSpace>> = Lazy::new(|| {
    Arc::new(RgbColorSpace::new(
        Point2f::new(0.64, 0.33),
        Point2f::new(0.3, 0.6),
        Point2f::new(0.15, 0.06),
        Spectrum::get_named_spectrum(NamedSpectrum::StdIllumD65).as_ref(),
        Gamut::SRgb,
    ))
});

pub static REC_2020: Lazy<Arc<RgbColorSpace>> = Lazy::new(|| {
    Arc::new(RgbColorSpace::new(
        Point2f::new(0.708, 0.292),
        Point2f::new(0.170, 0.797),
        Point2f::new(0.131, 0.046),
        Spectrum::get_named_spectrum(NamedSpectrum::StdIllumD65).as_ref(),
        Gamut::Rec2020,
    ))
});

pub static ACES2065_1: Lazy<Arc<RgbColorSpace>> = Lazy::new(|| {
    Arc::new(RgbColorSpace::new(
        Point2f::new(0.7347, 0.2653),
        Point2f::new(0.0, 1.0),
        Point2f::new(0.0001, -0.077),
        Spectrum::get_named_spectrum(NamedSpectrum::IllumAcesD60).as_ref(),
        Gamut::Aces2065_1,
    ))
});
