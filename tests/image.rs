use std::{path::PathBuf, sync::Arc};

use cray::{color::rgb_xyz::{ColorEncoding, ColorEncodingPtr, LinearColorEncoding}, image::{Image, ImageChannelValues, ImageMetadata, PixelFormat}, math::{Float, Point2i}};


#[test]
fn test_image_png() {
    let mut im = Image::new(PixelFormat::Float16, Point2i::new(256, 256), &[
        "R".to_string(),
        "G".to_string(),
        "B".to_string(),
    ], Some(ColorEncodingPtr(Arc::new(ColorEncoding::Linear(LinearColorEncoding)))));

    for x in 0..256 {
        for y in 0..256 {
            let r = x as Float / im.resolution().x as Float;
            let b = y as Float / im.resolution().y as Float;
            let values = ImageChannelValues { values: vec![r, 0.0, b].into_iter().collect() };
            im.set_channels(Point2i::new(x, y), &values);
        }
    }

    im.write(&PathBuf::from("output/image-test.png"), &ImageMetadata::default()).unwrap();
}
