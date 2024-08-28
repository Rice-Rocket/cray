use std::{collections::HashMap, ffi::OsStr, fmt::Display, fs::File, io, ops::{Index, IndexMut}, path::PathBuf, sync::Arc};

use arrayvec::ArrayVec;
use half::f16;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use tracing::warn;

use crate::{color::{colorspace::{NamedColorSpace, RgbColorSpace}, rgb_xyz::{AbstractColorEncoding as _, ColorEncoding, ColorEncodingPtr}}, modulo, reader::utils::truncate_filename, tile::Tile, vec2d::Vec2D, windowed_sinc, Bounds2f, Bounds2i, Float, Mat4, Point2f, Point2i};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PixelFormat {
    Uint256,
    Float16,
    Float32,
}

impl PixelFormat {
    pub fn is_8bit(&self) -> bool {
        *self == PixelFormat::Uint256
    }

    pub fn is_16bit(&self) -> bool {
        *self == PixelFormat::Float16
    }

    pub fn is_32bit(&self) -> bool {
        *self == PixelFormat::Float32
    }
}

impl Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PixelFormat::Uint256 => write!(f, "U256"),
            PixelFormat::Float16 => write!(f, "Half"),
            PixelFormat::Float32 => write!(f, "Float"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct ResampleWeight {
    first_pixel: i32,
    weight: [Float; 4]
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

impl WrapMode {
    const CLAMP_STR: &'static str = "clamp";
    const REPEAT_STR: &'static str = "repeat";
    const BLACK_STR: &'static str = "black";
    const OCTAHEDRAL_SPHERE_STR: &'static str = "octahedralsphere";

    pub fn parse(str: &str) -> Option<Self> {
        match str {
            WrapMode::CLAMP_STR => Some(WrapMode::Clamp),
            WrapMode::REPEAT_STR => Some(WrapMode::Repeat),
            WrapMode::BLACK_STR => Some(WrapMode::Black),
            WrapMode::OCTAHEDRAL_SPHERE_STR => Some(WrapMode::OctahedralSphere),
            _ => None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrapMode2D {
    x: WrapMode,
    y: WrapMode,
}

impl WrapMode2D {
    pub fn new(x: WrapMode, y: WrapMode) -> WrapMode2D {
        Self { x, y }
    }
    
    pub fn remap(self, mut p: Point2i, resolution: Point2i) -> Option<Point2i> {
        if self.x == WrapMode::OctahedralSphere {
            debug_assert!(self.y == WrapMode::OctahedralSphere);

            if p.x < 0 {
                p.x = -p.x;
                p.y = resolution.y - 1 - p.y;
            } else if p.x >= resolution.x {
                p.x = 2 * resolution.x - 1 - p.x;
                p.y = resolution.y - 1 - p.y;
            }

            if p.y < 0 {
                p.x = resolution.x - 1 - p.x;
                p.y = -p.y;
            } else if p.y >= resolution.y {
                p.x = resolution.x - 1 - p.x;
                p.y = 2 * resolution.y - 1 - p.y;
            }

            if resolution.x == 1 {
                p.x = 0;
            }

            if resolution.y == 1 {
                p.y = 0;
            }

            return Some(p);
        }

        for (c, wrap) in [self.x, self.y].into_iter().enumerate() {
            if p[c] >= 0 && p[c] < resolution[c] {
                continue;
            }

            match wrap {
                WrapMode::Black => return None,
                WrapMode::Clamp => p[c] = p[c].clamp(0, resolution[c] - 1),
                WrapMode::Repeat => p[c] = modulo(p[c], resolution[c]),
                WrapMode::OctahedralSphere => unreachable!(),
            }
        }

        Some(p)
    }
}

impl From<WrapMode> for WrapMode2D {
    fn from(value: WrapMode) -> Self {
        Self { x: value, y: value }
    }
}

impl Display for WrapMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WrapMode::Black => write!(f, "{}", WrapMode::BLACK_STR),
            WrapMode::Clamp => write!(f, "{}", WrapMode::CLAMP_STR),
            WrapMode::Repeat => write!(f, "{}", WrapMode::REPEAT_STR),
            WrapMode::OctahedralSphere => write!(f, "{}", WrapMode::OCTAHEDRAL_SPHERE_STR),
        }
    }
}

#[derive(Default)]
pub struct ImageMetadata {
    pub render_time_seconds: Option<Float>,
    pub camera_from_world: Option<Mat4>,
    pub ndc_from_world: Option<Mat4>,
    pub pixel_bounds: Option<Bounds2i>,
    pub full_resolution: Option<Point2i>,
    pub samples_per_pixel: Option<i32>,
    pub mse: Option<Float>,
    pub color_space: Option<Arc<RgbColorSpace>>,
    pub strings: HashMap<String, String>,
    pub string_vecs: HashMap<String, Vec<String>>,
}

pub struct ImageAndMetadata {
    pub image: Image,
    pub metadata: ImageMetadata,
}

#[derive(Default)]
pub struct ImageChannelDesc {
    offset: ArrayVec<i32, 4>,
}

impl ImageChannelDesc {
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn size(&self) -> usize {
        self.offset.len()
    }

    pub fn is_identity(&self) -> bool {
        for i in 0..self.offset.len() {
            if self.offset[i] != i as i32 {
                return false;
            }
        }

        true
    }
}

#[derive(Default)]
pub struct ImageChannelValues {
    pub values: ArrayVec<Float, 4>,
}

impl ImageChannelValues {
    pub fn new(size: usize, v: Float) -> Self {
        Self { values: vec![v; size].into_iter().collect() }
    }

    pub fn max_value(&self) -> Float {
        *self.values.iter().max_by(|a, b| a.partial_cmp(b).expect("unexpected NaN")).expect("tried to find max value of empty vector")
    }

    pub fn average(&self) -> Float {
        self.values.iter().sum::<Float>() / self.values.len() as Float
    }
}

impl Index<usize> for ImageChannelValues {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for ImageChannelValues {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl From<ImageChannelValues> for Float {
    fn from(value: ImageChannelValues) -> Self {
        debug_assert!(value.values.len() == 1);
        value[0]
    }
}

impl From<ImageChannelValues> for [Float; 3] {
    fn from(value: ImageChannelValues) -> Self {
        debug_assert!(value.values.len() == 3);
        [value[0], value[1], value[2]]
    }
}

#[derive(Debug, Clone)]
pub enum ImageData {
    Uint256(Vec<u8>),
    Float16(Vec<f16>),
    Float32(Vec<f32>),
}

impl ImageData {
    pub fn new(format: PixelFormat) -> ImageData {
        match format {
            PixelFormat::Uint256 => Self::Uint256(Vec::new()),
            PixelFormat::Float16 => Self::Float16(Vec::new()),
            PixelFormat::Float32 => Self::Float32(Vec::new()),
        }
    }

    pub fn format(&self) -> PixelFormat {
        match self {
            Self::Uint256(_) => PixelFormat::Uint256,
            Self::Float16(_) => PixelFormat::Float16,
            Self::Float32(_) => PixelFormat::Float32,
        }
    }

    pub fn as_u8(&self) -> &[u8] {
        if let ImageData::Uint256(ref data) = self {
            data
        } else {
            panic!("assumed u8 image data when it wasn't")
        }
    }
    
    pub fn as_f16(&self) -> &[f16] {
        if let ImageData::Float16(ref data) = self {
            data
        } else {
            panic!("assumed f16 image data when it wasn't")
        }
    }

    pub fn as_f32(&self) -> &[f32] {
        if let ImageData::Float32(ref data) = self {
            data
        } else {
            panic!("assumed f32 image data when it wasn't")
        }
    }

    pub fn as_u8_mut(&mut self) -> &mut [u8] {
        if let ImageData::Uint256(ref mut data) = self {
            data
        } else {
            panic!("assumed u8 image data when it wasn't")
        }
    }
    
    pub fn as_f16_mut(&mut self) -> &mut [f16] {
        if let ImageData::Float16(ref mut data) = self {
            data
        } else {
            panic!("assumed f16 image data when it wasn't")
        }
    }

    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        if let ImageData::Float32(ref mut data) = self {
            data
        } else {
            panic!("assumed f32 image data when it wasn't")
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    resolution: Point2i,
    channel_names: Vec<String>,
    color_encoding: Option<ColorEncodingPtr>,
    data: ImageData,
}

impl Image {
    pub fn new(
        format: PixelFormat,
        resolution: Point2i,
        channels: &[String],
        encoding: Option<ColorEncodingPtr>
    ) -> Image {
        let mut image = Image {
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: encoding,
            data: ImageData::new(format),
        };

        let n_channels = image.n_channels();
        match &mut image.data {
            ImageData::Uint256(ref mut data) => {
                data.resize(n_channels * resolution.x as usize * resolution.y as usize, 0);
                debug_assert!(image.color_encoding.is_some())
            },
            ImageData::Float16(ref mut data) => data.resize(n_channels * resolution.x as usize * resolution.y as usize, f16::from_f32(0.0)),
            ImageData::Float32(ref mut data) => data.resize(n_channels * resolution.x as usize * resolution.y as usize, 0.0),
        }

        image
    }

    pub fn new_u256(
        data: Vec<u8>,
        resolution: Point2i,
        channels: &[String],
        encoding: ColorEncodingPtr,
    ) -> Image {
        Image {
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: Some(encoding),
            data: ImageData::Uint256(data),
        }
    }

    pub fn select_channels(&self, desc: &ImageChannelDesc) -> Image {
        let desc_channel_names = desc.offset.iter().map(|i| self.channel_names[*i as usize].clone()).collect::<Vec<String>>();
        let mut image = Image::new(self.data.format(), self.resolution, &desc_channel_names, self.color_encoding.clone());

        for y in 0..self.resolution.y {
            for x in 0..self.resolution.x {
                let p = Point2i::new(x, y);
                let values = self.get_channels_from_desc(p, desc);
                image.set_channels(p, &values);
            }
        }

        image
    }

    pub fn for_extent<F: FnMut(&mut Self, usize)>(&mut self, extent: &Bounds2i, wrap_mode: WrapMode2D, mut op: F) {
        assert!(extent.min.x < extent.max.x);
        assert!(extent.min.y < extent.max.y);

        let nx = extent.max.x - extent.min.x;
        let nc = self.n_channels();
        let intersection = extent.intersect(Bounds2i::new(Point2i::ZERO, self.resolution));
        if !intersection.is_empty() && intersection == *extent {
            for y in extent.min.y..extent.max.y {
                let mut offset = self.pixel_offset(Point2i::new(extent.min.x, y));
                for _x in 0..nx {
                    for _c in 0..nc {
                        op(self, offset);
                        offset += 1;
                    }
                }
            }
        } else {
            for y in extent.min.y..extent.max.y {
                for x in 0..nx {
                    let p = Point2i::new(extent.min.x + x, y);
                    assert!(wrap_mode.remap(p, self.resolution).is_some());
                    let mut offset = self.pixel_offset(p);
                    for _c in 0..nc {
                        op(self, offset);
                        offset += 1;
                    }
                }
            }
        }
    }

    pub fn copy_rect_in(&mut self, extent: &Bounds2i, buf: &[f32]) {
        assert!(buf.len() >= extent.area() as usize * self.n_channels());
        let mut buf_offset = 0;

        match self.data.format() {
            PixelFormat::Uint256 => {
                let intersect = extent.intersect(Bounds2i::new(Point2i::ZERO, self.resolution));
                if !intersect.is_empty() && intersect == *extent {
                    let count = self.n_channels() * (extent.max.x - extent.min.x) as usize;
                    for y in extent.min.y..extent.max.y {
                        let offset = self.pixel_offset(Point2i::new(extent.min.x, y));
                        #[cfg(use_f64)]
                        {
                            for i in 0..count {
                                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.from_linear(
                                    &buf[buf_offset..buf_offset + 1],
                                    &mut self.data.as_u8_mut()[offset + i..offset + i + 1],
                                );
                                buf_offset += 1;
                            }
                        }
                        #[cfg(not(use_f64))]
                        {
                            self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.from_linear(
                                &buf[buf_offset..buf_offset + count],
                                &mut self.data.as_u8_mut()[offset..offset + count]
                            );
                        }
                        buf_offset += count;
                    }
                } else {
                    self.for_extent(extent, WrapMode::Clamp.into(), |image, offset: usize| {
                        image.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.from_linear(
                            &buf[buf_offset..buf_offset + 1],
                            &mut image.data.as_u8_mut()[offset..offset + 1],
                        );
                        buf_offset += 1;
                    })
                }
            },
            PixelFormat::Float16 => {
                self.for_extent(extent, WrapMode::Clamp.into(), |image, offset: usize| {
                    image.data.as_f16_mut()[offset] = f16::from_f32(buf[buf_offset]);
                    buf_offset += 1;
                })
            },
            PixelFormat::Float32 => {
                self.for_extent(extent, WrapMode::Clamp.into(), |image, offset: usize| {
                    image.data.as_f32_mut()[offset] = buf[buf_offset];
                    buf_offset += 1;
                })
            },
        }
    }

    pub fn copy_rect_out(&mut self, extent: &Bounds2i, buf: &mut [f32], wrap_mode: WrapMode2D) {
        assert!(buf.len() >= extent.area() as usize * self.n_channels());
        let mut buf_offset = 0;
        match self.data.format() {
            PixelFormat::Uint256 => {
                let intersect = extent.intersect(Bounds2i::new(Point2i::ZERO, self.resolution));
                if !intersect.is_empty() && intersect == *extent {
                    let count = self.n_channels() * (extent.max.x - extent.min.x) as usize;
                    for y in extent.min.y..extent.max.y {
                        let offset = self.pixel_offset(Point2i::new(extent.min.x, y));
                        #[cfg(use_f64)]
                        {
                            for i in 0..count {
                                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.to_linear(
                                    &self.data.as_u8()[offset + i..offset + i + 1],
                                    &mut buf[buf_offset..buf_offset + 1],
                                );
                                buf_offset += 1;
                            }
                        }
                        #[cfg(not(use_f64))]
                        {
                            self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.to_linear(
                                &self.data.as_u8()[offset..offset + count],
                                &mut buf[buf_offset..buf_offset + count],
                                );
                            buf_offset += count;
                        }
                    }
                } else {
                    self.for_extent(extent, wrap_mode, |image, offset: usize| {
                        image.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0.to_linear(
                            &image.data.as_u8()[offset..offset + 1],
                            &mut buf[buf_offset..buf_offset + 1],
                        );
                        buf_offset += 1;
                    })
                }
            },
            PixelFormat::Float16 => {
                self.for_extent(extent, wrap_mode, |image, offset: usize| {
                    buf[buf_offset] = image.data.as_f16()[offset].to_f32();
                    buf_offset += 1;
                })
            },
            PixelFormat::Float32 => {
                self.for_extent(extent, wrap_mode, |image, offset: usize| {
                    buf[buf_offset] = image.data.as_f32()[offset];
                    buf_offset += 1;
                })
            },
        }
    }

    pub fn convert_to_format(&self, format: PixelFormat) -> Image {
        if self.data.format() == format {
            return self.clone();
        }

        let mut new_image = Image::new(
            format,
            self.resolution,
            &self.channel_names,
            self.color_encoding.clone(),
        );

        for y in 0..self.resolution.y {
            for x in 0..self.resolution.x {
                for c in 0..self.n_channels() {
                    let v = self.get_channel(Point2i::new(x, y), c);
                    new_image.set_channel(Point2i::new(x, y), c, v);
                }
            }
        }

        new_image
    }

    pub fn float_resize_up(&mut self, new_res: Point2i, wrap_mode: WrapMode2D) -> Image {
        assert!(new_res.x > self.resolution.x);
        assert!(new_res.y > self.resolution.y);

        let mut resampled_image = Image::new(PixelFormat::Float32, new_res, &self.channel_names, None);

        let x_weights = self.resample_weights(self.resolution.x as usize, new_res.x as usize);
        let y_weights = self.resample_weights(self.resolution.y as usize, new_res.y as usize);

        let tiles = Tile::tiles(Bounds2i::new(Point2i::ZERO, new_res), 8, 8);
        tiles.into_iter().for_each(|tile| {
            let in_extent = Bounds2i::new(
                Point2i::new(
                    x_weights[tile.bounds.min.x as usize].first_pixel,
                    y_weights[tile.bounds.min.y as usize].first_pixel,
                ),
                Point2i::new(
                    x_weights[tile.bounds.max.x as usize - 1].first_pixel + 4,
                    y_weights[tile.bounds.max.y as usize - 1].first_pixel + 4,
                ),
            );

            let mut in_buf = vec![0.0; in_extent.area() as usize * self.n_channels()];
            self.copy_rect_out(&in_extent, &mut in_buf, wrap_mode);

            let nx_out = tile.bounds.max.x - tile.bounds.min.x;
            let ny_out = tile.bounds.max.y - tile.bounds.min.y;
            let nx_in = in_extent.max.x - in_extent.min.x;
            let ny_in = in_extent.max.y - in_extent.min.y;

            let mut x_buf = vec![0.0; (ny_in * nx_out) as usize * self.n_channels()];
            let mut x_buf_offset = 0;

            for y_out in in_extent.min.y..in_extent.max.y {
                for x_out in tile.bounds.min.x..tile.bounds.max.x {
                    debug_assert!(x_out >= 0 && x_out < x_weights.len() as i32);
                    let rsw = &x_weights[x_out as usize];
                    let x_in = rsw.first_pixel - in_extent.min.x;
                    debug_assert!(x_in >= 0);
                    debug_assert!(x_in + 3 < nx_in);
                    let y_in = y_out - in_extent.min.y;
                    let mut in_offset = self.n_channels() * (x_in + y_in * nx_in) as usize;
                    debug_assert!(in_offset + 3 * self.n_channels() < in_buf.len());

                    for _c in 0..self.n_channels() {
                        x_buf[x_buf_offset] = rsw.weight[0] * in_buf[in_offset]
                            + rsw.weight[1] * in_buf[in_offset + self.n_channels()]
                            + rsw.weight[2] * in_buf[in_offset + 2 * self.n_channels()]
                            + rsw.weight[3] * in_buf[in_offset + 3 * self.n_channels()];
                        x_buf_offset += 1;
                        in_offset += 1;
                    }
                }
            }

            let mut out_buf = vec![0.0; (nx_out * ny_out) as usize * self.n_channels()];
            for x in 0..nx_out {
                for y in 0..ny_out {
                    let y_out = y + tile.bounds[0][1];
                    debug_assert!(y_out >= 0);
                    debug_assert!(y_out < y_weights.len() as i32);
                    let rsw = &y_weights[y_out as usize];

                    debug_assert!(rsw.first_pixel - in_extent[0][1] >= 0);
                    let mut x_buf_offset = self.n_channels() * (x + nx_out * (rsw.first_pixel - in_extent[0][1])) as usize;
                    let step = self.n_channels() * nx_out as usize;
                    debug_assert!(x_buf_offset + 3 * step < x_buf.len());

                    let mut out_offset = self.n_channels() * (x + y * nx_out) as usize;
                    for _c in 0..self.n_channels() {
                        out_buf[out_offset] = Float::max(0.0, rsw.weight[0] * x_buf[x_buf_offset]
                            + rsw.weight[1] * x_buf[x_buf_offset + step]
                            + rsw.weight[2] * x_buf[x_buf_offset + 2 * step]
                            + rsw.weight[3] * x_buf[x_buf_offset + 3 * step]);

                        out_offset += 1;
                        x_buf_offset += 1;
                    }
                }
            }

            resampled_image.copy_rect_in(&tile.bounds, &out_buf);
        });

        resampled_image
    }

    fn resample_weights(&self, old_res: usize, new_res: usize) -> Vec<ResampleWeight> {
        assert!(old_res < new_res);
        let mut wt = vec![ResampleWeight::default(); new_res];
        let filter_radius = 2.0;
        let tau = 2.0;
        for (i, w) in wt.iter_mut().enumerate().take(new_res) {
            let center = (i as Float + 0.5) * old_res as Float / new_res as Float;
            w.first_pixel = ((center - filter_radius + 0.5).floor() as i32).max(0);
            for j in 0..4 {
                let pos = w.first_pixel as Float + 0.5;
                w.weight[j] = windowed_sinc(pos - center, filter_radius, tau);
            }

            let inv_sum_wts = 1.0 / (w.weight[0] + w.weight[1] + w.weight[2] + w.weight[3]);
            for j in 0..4 {
                w.weight[j] *= inv_sum_wts;
            }
        }
        for i in 0..new_res {
        }

        wt
    }

    pub fn generate_pyramid(mut image: Image, wrap_mode: WrapMode2D) -> Vec<Image> {
        let orig_format = image.data.format();
        let n_channels = image.n_channels();
        let orig_encoding = image.color_encoding.clone();
        
        let mut image = if !(image.resolution.x as u32).is_power_of_two() || !(image.resolution.y as u32).is_power_of_two() {
            image.float_resize_up(Point2i::new(
                (image.resolution.x as u32).next_power_of_two() as i32,
                (image.resolution.y as u32).next_power_of_two() as i32,
            ), wrap_mode)
        } else {
            image.convert_to_format(PixelFormat::Float32)
        };

        assert!(image.data.format().is_32bit());

        let n_levels = 1 + ((i32::max(image.resolution.x, image.resolution.y) as Float).log2() as i32);
        let mut pyramid = Vec::with_capacity(n_levels as usize);

        for i in 0..(n_levels - 1) {
            pyramid.push(Image::new(orig_format, image.resolution, &image.channel_names, orig_encoding.clone()));

            let next_resolution = Point2i::new(
                i32::max(1, (image.resolution.x + 1) / 2),
                i32::max(1, (image.resolution.y + 1) / 2),
            );
            let mut next_image = Image::new(image.data.format(), next_resolution, &image.channel_names, orig_encoding.clone());

            let src_deltas = [
                0,
                n_channels,
                n_channels * image.resolution.x as usize,
                n_channels * (image.resolution.x as usize + 1),
            ];

            (0..next_resolution.y).for_each(|y| {
                let mut src_offset = image.pixel_offset(Point2i::new(0, 2 * y));
                let mut next_offset = next_image.pixel_offset(Point2i::new(0, y));

                for _x in 0..next_resolution.x {
                    for _c in 0..n_channels {
                        next_image.data.as_f32_mut()[next_offset] = (
                            image.data.as_f32()[src_offset]
                            + image.data.as_f32()[src_offset + src_deltas[1]]
                            + image.data.as_f32()[src_offset + src_deltas[2]]
                            + image.data.as_f32()[src_offset + src_deltas[3]]
                        ) * 0.25;

                        src_offset += 1;
                        next_offset += 1;
                    }
                    src_offset += n_channels;
                }

                let y_start = 2 * y;
                let y_end = i32::min(2 * y + 2, image.resolution.y);
                let offset = image.pixel_offset(Point2i::new(0, y_start));
                let count = (y_end - y_start) * n_channels as i32 * image.resolution.x;
                if let ImageData::Float32(ref data) = &image.data {
                    pyramid[i as usize].copy_rect_in(
                        &Bounds2i::new(Point2i::new(0, y_start), Point2i::new(image.resolution.x, y_end)),
                        &data[offset..offset + count as usize],
                    );
                }
            });

            image = next_image;
        }

        assert!(image.resolution.x == 1 && image.resolution.y == 1);
        pyramid.push(Image::new(orig_format, Point2i::new(1, 1), &image.channel_names, orig_encoding));
        if let ImageData::Float32(ref data) = &image.data {
            pyramid[n_levels as usize - 1].copy_rect_in(
                &Bounds2i::new(Point2i::new(0, 0), Point2i::new(1, 1)),
                &data[0..n_channels],
            );
        }

        pyramid
    }

    pub fn resolution(&self) -> Point2i {
        self.resolution
    }

    pub fn n_channels(&self) -> usize {
        self.channel_names.len()
    }

    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }

    pub fn encoding(&self) -> Option<&ColorEncodingPtr> {
        self.color_encoding.as_ref()
    }

    pub fn is_empty(&self) -> bool {
        self.resolution.x > 0 && self.resolution.y < 0
    }
    
    pub fn bytes_used(&self) -> usize {
        match &self.data {
            ImageData::Uint256(ref data) => data.len(),
            ImageData::Float16(ref data) => 2 * data.len(),
            ImageData::Float32(ref data) => 4 * data.len(),
        }
    }

    pub fn pixel_offset(&self, p: Point2i) -> usize {
        debug_assert!(Bounds2i::new(Point2i::ZERO, self.resolution).inside_exclusive(p));
        (self.n_channels() as i32 * (p.y * self.resolution.x + p.x)) as usize
    }

    pub fn get_channel(&self, p: Point2i, c: usize) -> Float {
        self.get_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    pub fn get_channel_wrapped(&self, p: Point2i, c: usize, wrap_mode: WrapMode2D) -> Float {
        if let Some(p) = wrap_mode.remap(p, self.resolution) {
            match &self.data {
                ImageData::Uint256(ref data) => {
                    let mut r = [0.0];
                    self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                        .to_linear(&[data[self.pixel_offset(p) + c]], &mut r);
                    r[0]
                },
                ImageData::Float16(ref data) => {
                    data[self.pixel_offset(p) + c].into()
                },
                ImageData::Float32(ref data) => {
                    data[self.pixel_offset(p) + c]
                },
            }
        } else {
            0.0
        }
    }

    pub fn get_channels(&self, p: Point2i) -> ImageChannelValues {
        self.get_channels_wrapped(p, WrapMode::Clamp.into())
    }

    pub fn get_channels_wrapped(&self, p: Point2i, wrap_mode: WrapMode2D) -> ImageChannelValues {
        let mut cv = ImageChannelValues::new(self.n_channels(), 0.0);
        let Some(p) = wrap_mode.remap(p, self.resolution) else { return cv };

        let pixel_offset = self.pixel_offset(p);
        match &self.data {
            ImageData::Uint256(ref data) => {
                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                    .to_linear(&data[pixel_offset..pixel_offset + self.n_channels()], &mut cv.values)
            },
            ImageData::Float16(ref data) => {
                for i in 0..self.n_channels() {
                    cv[i] = data[pixel_offset + i].into();
                }
            },
            ImageData::Float32(ref data) => {
                for i in 0..self.n_channels() {
                    cv[i] = data[pixel_offset + i];
                }
            },
        }

        cv
    }

    pub fn get_channels_from_desc(&self, p: Point2i, desc: &ImageChannelDesc) -> ImageChannelValues {
        self.get_channels_from_desc_wrapped(p, desc, WrapMode::Clamp.into())
    }

    pub fn get_channels_from_desc_wrapped(
        &self, 
        p: Point2i,
        desc: &ImageChannelDesc,
        wrap_mode: WrapMode2D
    ) -> ImageChannelValues {
        let mut cv = ImageChannelValues::new(desc.offset.len(), 0.0);
        let Some(p) = wrap_mode.remap(p, self.resolution) else { return cv };

        let pixel_offset = self.pixel_offset(p);

        match &self.data {
            ImageData::Uint256(ref data) => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                        .to_linear(&data[index..index + 1], &mut cv.values[i..i + 1]);
                }
            },
            ImageData::Float16(ref data) => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    cv[i] = data[index].to_f32();
                }
            },
            ImageData::Float32(ref data) => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    cv[i] = data[index];
                }
            },
        }

        cv
    }

    pub fn lookup_nearest_channel(&self, p: Point2f, c: usize) -> Float {
        self.lookup_nearest_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    pub fn lookup_nearest_channel_wrapped(&self, p: Point2f, c: usize, wrap_mode: WrapMode2D) -> Float {
        let pi = Point2i::new(
            (p.x * self.resolution.x as Float) as i32,
            (p.y * self.resolution.y as Float) as i32,
        );

        self.get_channel_wrapped(pi, c, wrap_mode)
    }

    pub fn bilinear(&self, p: Point2f, wrap_mode: WrapMode2D) -> ImageChannelValues
    {
        let mut cv = ImageChannelValues::new(self.n_channels(), 0.0);
        for c in 0..self.n_channels()
        {
            cv[c] = self.bilinear_channel_wrapped(p, c, wrap_mode);
        }
        cv
    }

    pub fn bilinear_channels_wrapped(&self, p: Point2f, wrap_mode: WrapMode2D) -> ImageChannelValues {
        let mut cv = ImageChannelValues::new(self.n_channels(), 0.0);
        for c in 0..self.n_channels() {
            cv[c] = self.bilinear_channel_wrapped(p, c, wrap_mode);
        }

        cv
    }

    pub fn bilinear_channel(&self, p: Point2f, c: usize) -> Float {
        self.bilinear_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    pub fn bilinear_channel_wrapped(&self, p: Point2f, c: usize, wrap_mode: WrapMode2D) -> Float {
        let x = p.x * self.resolution.x as Float - 0.5;
        let y = p.y * self.resolution.y as Float - 0.5;
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let dx = x - xi as Float;
        let dy = y - yi as Float;

        let v: [Float; 4] = [
            self.get_channel_wrapped(Point2i::new(xi, yi), c, wrap_mode),
            self.get_channel_wrapped(Point2i::new(xi + 1, yi), c, wrap_mode),
            self.get_channel_wrapped(Point2i::new(xi, yi + 1), c, wrap_mode),
            self.get_channel_wrapped(Point2i::new(xi + 1, yi + 1), c, wrap_mode),
        ];

        (1.0 - dx) * (1.0 - dy) * v[0]
            + dx * (1.0 - dy) * v[1]
            + (1.0 - dx) * dy * v[2]
            + dx * dy * v[3]
    }

    pub fn set_channel(&mut self, p: Point2i, c: usize, value: Float) {
        let value = if value.is_nan() { warn!("tried to store NaN value in image, using 0.0 instead"); 0.0 } else { value };

        let index = self.pixel_offset(p) + c;
        match &mut self.data {
            ImageData::Uint256(ref mut data) => {
                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                    .from_linear(&[value], &mut data[index..index + 1])
            },
            ImageData::Float16(ref mut data) => data[index] = f16::from_f32(value),
            ImageData::Float32(ref mut data) => data[index] = value,
        }
    }

    pub fn set_channels(&mut self, p: Point2i, values: &ImageChannelValues) {
        assert!(values.values.len() == self.n_channels());
        let index = self.pixel_offset(p);

        match &mut self.data {
            ImageData::Uint256(ref mut data) => {
                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                    .from_linear(&values.values, &mut data[index..index + values.values.len()])
            },
            ImageData::Float16(ref mut data) => {
                for i in 0..values.values.len() {
                    data[index + i] = f16::from_f32(values[i]);
                }
            },
            ImageData::Float32(ref mut data) => {
                for i in 0..values.values.len() {
                    data[index + i] = values[i];
                }
            },
        }
    }

    pub fn set_channels_slice(&mut self, p: Point2i, values: &[Float]) {
        assert!(values.len() == self.n_channels());
        let index = self.pixel_offset(p);

        match &mut self.data {
            ImageData::Uint256(ref mut data) => {
                self.color_encoding.as_ref().expect("U256 pixel format requires color encoding").0
                    .from_linear(values, &mut data[index..index + values.len()])
            },
            ImageData::Float16(ref mut data) => {
                for i in 0..values.len() {
                    data[index + i] = f16::from_f32(values[i]);
                }
            },
            ImageData::Float32(ref mut data) => {
                data[index..(values.len() + index)].copy_from_slice(values);
            },
        }
    }

    pub fn get_channel_desc(&self, channels: &[&str]) -> Option<ImageChannelDesc> {
        let mut offset: ArrayVec<i32, 4> = [0; 4].into();

        for i in 0..channels.len() {
            let mut j = 0;
            while j < self.channel_names.len() {
                if channels[i] == self.channel_names[j] {
                    offset[i] = j as i32;
                    break;
                }
                j += 1;
            }

            if j == self.channel_names.len() {
                return None;
            }
        }

        offset.truncate(channels.len());
        Some(ImageChannelDesc { offset })
    }

    pub fn all_channels_desc(&self) -> ImageChannelDesc {
        let mut offset = ArrayVec::<i32, 4>::new();

        for i in 0..self.n_channels() {
            offset[i] = i as i32;
        }

        ImageChannelDesc { offset }
    }

    pub fn get_default_sampling_distribution(&self) -> Vec2D<Float> {
        self.get_sampling_distribution(|_p| 1.0, &Bounds2f::from_point(Point2f::ONE))
    }

    pub fn get_sampling_distribution<F: Fn(Point2f) -> Float>(&self, dx_da: F, domain: &Bounds2f) -> Vec2D<Float> {
        let mut dist = Vec2D::from_bounds(Bounds2i::new(Point2i::ZERO, self.resolution));

        for y in 0..self.resolution.y {
            for x in 0..self.resolution.x {
                let value = self.get_channels(Point2i::new(x, y)).average();
                let p = domain.lerp(Point2f::new((x as Float + 0.5) / self.resolution.x as Float,
                    (y as Float + 0.5) / self.resolution.y as Float));
                dist.set(Point2i::new(x, y), value * dx_da(p))
            }
        }

        dist
    }

    pub fn read(path: &PathBuf, encoding: Option<ColorEncodingPtr>) -> io::Result<ImageAndMetadata> {
        if path.extension().is_none() {
            return io::Result::Err(io::Error::new(io::ErrorKind::InvalidInput, format!("no file extension for {}", path.to_str().unwrap())))
        }

        if path.extension().unwrap().eq("png") {
            Self::read_png(path, encoding)
        } else if path.extension().unwrap().eq("exr") {
            Self::read_exr(path, encoding)
        } else {
            io::Result::Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unsupported file extension for {}", path.to_str().unwrap())))
        }
    }
    
    fn read_png(path: &PathBuf, encoding: Option<ColorEncodingPtr>) -> io::Result<ImageAndMetadata> {
        let encoding = if let Some(encoding) = encoding {
            encoding
        } else {
            ColorEncoding::get("srgb", None).clone()
        };

        let mut decoder = png::Decoder::new(File::open(path).unwrap());
        decoder.set_transformations(png::Transformations::IDENTITY);
        let mut reader = decoder.read_info().unwrap();

        let mut im_data = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut im_data).unwrap();

        let image = match info.color_type {
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
                let im_data = if info.color_type == png::ColorType::GrayscaleAlpha {
                    im_data.chunks_exact(2).map(|chunk| chunk[0]).collect::<Vec<u8>>()
                } else {
                    im_data
                };

                match info.bit_depth {
                    png::BitDepth::Eight => Image::new_u256(
                        im_data,
                        Point2i::new(info.width as i32, info.height as i32),
                        &["Y".to_owned()],
                        encoding
                    ),
                    png::BitDepth::Sixteen => {
                        let mut image = Image::new(
                            PixelFormat::Float16,
                            Point2i::new(info.width as i32, info.height as i32),
                            &["Y".to_owned()],
                            None,
                        );

                        for y in 0..info.height {
                            for x in 0..info.width {
                                let v = f16::from_le_bytes(
                                    im_data[(2 * (y * info.width + x)) as usize..(2 * (y * info.width + x) + 2) as usize]
                                        .try_into().unwrap()
                                );
                                let v: Float = v.into();
                                let v = encoding.0.to_float_linear(v);
                                image.set_channel(Point2i::new(x as i32, y as i32), 0, v);
                            }
                        }

                        image
                    },
                    _ => return io::Result::Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported bit depth"))
                }
            },
            png::ColorType::Rgb | png::ColorType::Rgba => {
                let has_alpha = info.color_type == png::ColorType::Rgba;
                match info.bit_depth {
                    png::BitDepth::Eight => match has_alpha {
                        true => Image::new_u256(
                            im_data,
                            Point2i::new(info.width as i32, info.height as i32),
                            &["R".to_owned(), "G".to_owned(), "B".to_owned(), "A".to_owned()],
                            encoding,
                        ),
                        false => Image::new_u256(
                            im_data,
                            Point2i::new(info.width as i32, info.height as i32),
                            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
                            encoding,
                        ),
                    },
                    png::BitDepth::Sixteen => match has_alpha {
                        true => {
                            let mut image = Image::new(
                                PixelFormat::Float16,
                                Point2i::new(info.width as i32, info.height as i32),
                                &["R".to_owned(), "G".to_owned(), "B".to_owned(), "A".to_owned()],
                                None,
                            );

                            let mut idx = 0;
                            for y in 0..info.height {
                                for x in 0..info.width {
                                    let r: Float = (((im_data[idx] as i32) << 8) + (im_data[idx + 1] as i32)) as Float / 65535.0;
                                    let g: Float = (((im_data[idx + 2] as i32) << 8) + (im_data[idx + 3] as i32)) as Float / 65535.0;
                                    let b: Float = (((im_data[idx + 4] as i32) << 8) + (im_data[idx + 5] as i32)) as Float / 65535.0;
                                    let a: Float = (((im_data[idx + 6] as i32) << 8) + (im_data[idx + 7] as i32)) as Float / 65535.0;
                                    let rgba = [r, g, b, a];
                                    for (i, c) in rgba.into_iter().enumerate() {
                                        let cv = encoding.0.to_float_linear(c);
                                        image.set_channel(Point2i::new(x as i32, y as i32), i, cv);
                                    }
                                    idx += 8;
                                }
                            }

                            image
                        },
                        false => {
                            let mut image = Image::new(
                                PixelFormat::Float16,
                                Point2i::new(info.width as i32, info.height as i32),
                                &["R".to_owned(), "G".to_owned(), "B".to_owned()],
                                None,
                            );

                            let mut idx = 0;
                            for y in 0..info.height {
                                for x in 0..info.width {
                                    let r: Float = (((im_data[idx] as i32) << 8) + (im_data[idx + 1] as i32)) as Float / 65535.0;
                                    let g: Float = (((im_data[idx + 2] as i32) << 8) + (im_data[idx + 3] as i32)) as Float / 65535.0;
                                    let b: Float = (((im_data[idx + 4] as i32) << 8) + (im_data[idx + 5] as i32)) as Float / 65535.0;
                                    let rgb = [r, g, b];
                                    for (i, c) in rgb.into_iter().enumerate() {
                                        let cv = encoding.0.to_float_linear(c);
                                        image.set_channel(Point2i::new(x as i32, y as i32), i, cv);
                                    }
                                    idx += 6;
                                }
                            }

                            image
                        }
                    },
                    _ => return io::Result::Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported bit depth")),
                }
            },
            png::ColorType::Indexed => return io::Result::Err(io::Error::new(io::ErrorKind::InvalidData, "indexed PNGs are not supported")),
        };

        let metadata = match info.color_type {
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => ImageMetadata::default(),
            png::ColorType::Rgb | png::ColorType::Rgba => {
                ImageMetadata {
                    color_space: Some(RgbColorSpace::get_named(NamedColorSpace::SRgb).clone()),
                    ..Default::default()
                }
            },
            png::ColorType::Indexed => unreachable!(),
        };

        io::Result::Ok(ImageAndMetadata { image, metadata })
    }

    fn read_exr(path: &PathBuf, encoding: Option<ColorEncodingPtr>) -> io::Result<ImageAndMetadata> {
        use exr::prelude::{ReadChannels, ReadLayers};

        let image_name = truncate_filename(path);

        let mut bar_template = format!("Reading image '{}' ", &image_name);
        bar_template += "{spinner:.green} [{elapsed}] [{bar:30.white/white}] {percent} ({eta})";
        let style = ProgressStyle::with_template(&bar_template).unwrap()
            .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        let mut bar = ProgressBar::new(1000).with_style(style);

        let Ok(im_exr) = exr::image::read::read()
            .no_deep_data()
            .largest_resolution_level() // or all_resolution_levels()
            .all_channels()
            .first_valid_layer() // or all_layers()
            .all_attributes()
            .on_progress(|progress: f64| bar.set_position((progress * 1000.0) as u64))
            .from_file(path) else { return io::Result::Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("image file {} not found", path.to_str().unwrap()),
            ))};

        bar.finish_and_clear();

        let attrs = &im_exr.attributes;
        let resolution = attrs.display_window.size;
        let mut channel_names = Vec::new();
        let mut pixel_format = PixelFormat::Float32;

        if im_exr.layer_data.channel_data.list.is_empty() {
            return io::Result::Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "EXR images with no channels not supported",
            ));
        }

        for channel in im_exr.layer_data.channel_data.list.iter() {
            let name = &channel.name;
            channel_names.push(name.to_string())
        }

        let channel_info = im_exr.layer_data.channel_data.list.first().unwrap();
        let sample_vec = &channel_info.sample_data;
        match sample_vec {
            exr::image::FlatSamples::F16(_) => pixel_format = PixelFormat::Float16,
            exr::image::FlatSamples::F32(_) => pixel_format = PixelFormat::Float32,
            exr::image::FlatSamples::U32(_) => return io::Result::Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported image data format: U32",
            )),
        }

        let mut image = Image::new(
            pixel_format,
            Point2i::new(resolution.x() as i32, resolution.y() as i32),
            &channel_names,
            None,
        );

        let mut bar_template = format!("Processing image '{}' ", &image_name);
        bar_template += "{spinner:.green} [{elapsed}] [{bar:30.white/white}] {percent} ({eta})";
        let style = ProgressStyle::with_template(&bar_template).unwrap()
            .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        let mut bar = ProgressBar::new(image.resolution.x as u64 * channel_names.len() as u64).with_style(style);

        for (i, channel) in im_exr.layer_data.channel_data.list.iter().enumerate() {
            for x in 0..image.resolution.x {
                for y in 0..image.resolution.y {
                    let v = channel.sample_data.value_by_flat_index((y * image.resolution.x + x) as usize);
                    image.set_channel(Point2i::new(x, y), i, v.to_f32());
                }
                bar.inc(1);
            }
        }

        bar.finish_and_clear();

        let color_space = if channel_names.contains(&"R".to_owned()) 
            && channel_names.contains(&"G".to_owned()) 
            && channel_names.contains(&"B".to_owned())
        {
            Some(RgbColorSpace::get_named(NamedColorSpace::SRgb).clone())
        } else {
            None
        };

        let metadata = ImageMetadata {
            color_space,
            ..Default::default()
        };

        io::Result::Ok(ImageAndMetadata { image, metadata })
    }

    pub fn write(&self, path: &PathBuf, metadata: &ImageMetadata) -> io::Result<()> {
        if path.extension().is_none() {
            return io::Result::Err(io::Error::new(io::ErrorKind::InvalidInput, format!("no file extension for {}", path.to_str().unwrap())))
        }

        if path.extension().unwrap().eq("png") {
            self.write_png(path, metadata)
        } else if path.extension().unwrap().eq("exr") {
            self.write_exr(path, metadata)
        } else {
            io::Result::Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unsupported file extension for {}", path.to_str().unwrap())))
        }
    }

    fn write_png(&self, path: &PathBuf, _metadata: &ImageMetadata) -> io::Result<()> {
        let file = File::create(path)?;
        let buf_writer = &mut io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(buf_writer, self.resolution.x as u32, self.resolution.y as u32);
        
        match self.n_channels() {
            1 => encoder.set_color(png::ColorType::Grayscale),
            2 => encoder.set_color(png::ColorType::GrayscaleAlpha),
            3 => encoder.set_color(png::ColorType::Rgb),
            4 => encoder.set_color(png::ColorType::Rgba),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{} color channels are not supported by PNG", self.n_channels())))
        }

        match self.data.format() {
            PixelFormat::Uint256 => encoder.set_depth(png::BitDepth::Eight),
            PixelFormat::Float16 => encoder.set_depth(png::BitDepth::Sixteen),
            PixelFormat::Float32 => return Err(io::Error::new(io::ErrorKind::InvalidData, "Float32 pixel format is not supported by PNG"))
        }

        let data = match &self.data {
            ImageData::Uint256(ref data) => data,
            ImageData::Float16(ref data) => &data.iter()
                .map(|v| v.to_f32().clamp(0.0, 1.0))
                .flat_map(|v| ((v * u16::MAX as f32) as u16).to_be_bytes()).collect::<Vec<u8>>(),
            _ => unreachable!(),
        };

        let mut writer = encoder.write_header().unwrap();
        Ok(writer.write_image_data(data)?)
    }

    fn write_exr(&self, path: &PathBuf, _metadata: &ImageMetadata) -> io::Result<()> {
        use exr::{image::{Encoding, Layer, SpecificChannels}, math::Vec2, prelude::{LayerAttributes, WritableImage}};

        match self.n_channels() {
            1 => unimplemented!(),
            2 => unimplemented!(),
            3 => {
                let get_pixels = |p: Vec2<usize>| {
                    let c = self.get_channels(Point2i::new(p.x() as i32, p.y() as i32));
                    (c.values[0], c.values[1], c.values[2])
                };

                let layer = Layer::new(
                    (self.resolution.x as usize, self.resolution.y as usize),
                    LayerAttributes::named("rgb main layer"),
                    Encoding::FAST_LOSSLESS,
                    SpecificChannels::rgb(get_pixels),
                );

                let mut image = exr::image::Image::from_layer(layer);
                image.attributes.pixel_aspect = 1.0;

                if let Err(e) = image.write().to_file(path) {
                    let errorkind = match e {
                        exr::error::Error::Aborted => io::ErrorKind::Interrupted,
                        exr::error::Error::NotSupported(_) => io::ErrorKind::InvalidInput,
                        exr::error::Error::Invalid(_) => io::ErrorKind::InvalidData,
                        exr::error::Error::Io(err) => return io::Result::Err(err),
                    };

                    io::Result::Err(io::Error::new(
                        errorkind,
                        "failed to write EXR image to file",
                    ))
                } else {
                    Ok(())
                }
            },
            4 => {
                let get_pixels = |p: Vec2<usize>| {
                    let c = self.get_channels(Point2i::new(p.x() as i32, p.y() as i32));
                    (c.values[0], c.values[1], c.values[2], c.values[3])
                };

                let layer = Layer::new(
                    (self.resolution.x as usize, self.resolution.y as usize),
                    LayerAttributes::named("rgba main layer"),
                    Encoding::FAST_LOSSLESS,
                    SpecificChannels::rgba(get_pixels),
                );

                let mut image = exr::image::Image::from_layer(layer);
                image.attributes.pixel_aspect = 1.0;

                if let Err(e) = image.write().to_file(path) {
                    let errorkind = match e {
                        exr::error::Error::Aborted => io::ErrorKind::Interrupted,
                        exr::error::Error::NotSupported(_) => io::ErrorKind::InvalidInput,
                        exr::error::Error::Invalid(_) => io::ErrorKind::InvalidData,
                        exr::error::Error::Io(err) => return io::Result::Err(err),
                    };

                    io::Result::Err(io::Error::new(
                        errorkind,
                        "failed to write EXR image to file",
                    ))
                } else {
                    Ok(())
                }
            },
            _ => io::Result::Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("cannot write {} image channels", self.n_channels()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{color::rgb_xyz::{ColorEncoding, ColorEncodingPtr, LinearColorEncoding}, image::Image, Point2i};

    #[test]
    fn image_basics() {
        let encoding = ColorEncodingPtr(Arc::new(ColorEncoding::Linear(LinearColorEncoding {})));
        let y8 = Image::new(
            super::PixelFormat::Uint256,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            Some(encoding.clone()),
        );
        assert_eq!(y8.n_channels(), 1);
        assert_eq!(
            y8.bytes_used(),
            (y8.resolution()[0] * y8.resolution()[1]) as usize
        );

        let y16 = Image::new(
            super::PixelFormat::Float16,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            None,
        );
        assert_eq!(y16.n_channels(), 1);
        assert_eq!(
            y16.bytes_used(),
            (2 * y16.resolution()[0] * y16.resolution()[1]) as usize
        );

        let y32 = Image::new(
            super::PixelFormat::Float32,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            None,
        );
        assert_eq!(y32.n_channels(), 1);
        assert_eq!(
            y32.bytes_used(),
            (4 * y32.resolution()[0] * y32.resolution()[1]) as usize
        );

        let rgb8 = Image::new(
            crate::image::PixelFormat::Uint256,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            Some(encoding),
        );
        assert_eq!(rgb8.n_channels(), 3);
        assert_eq!(
            rgb8.bytes_used(),
            (3 * rgb8.resolution()[0] * rgb8.resolution()[1]) as usize
        );

        let rgb16 = Image::new(
            crate::image::PixelFormat::Float16,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            None,
        );
        assert_eq!(rgb16.n_channels(), 3);
        assert_eq!(
            rgb16.bytes_used(),
            (2 * 3 * rgb16.resolution()[0] * rgb16.resolution()[1]) as usize
        );

        let rgb32 = Image::new(
            crate::image::PixelFormat::Float32,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            None,
        );
        assert_eq!(rgb32.n_channels(), 3);
        assert_eq!(
            rgb32.bytes_used(),
            (4 * 3 * rgb32.resolution()[0] * rgb32.resolution()[1]) as usize
        );
    }
}
