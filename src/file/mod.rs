use std::fs;

use crate::{options::Options, warn, Float};

pub fn read_float_file(filename: &str) -> Vec<Float> {
    let contents = fs::read_to_string(filename);
    match contents {
        Ok(s) => s
            .split_whitespace()
            .map(|t| {
                let float_val: Float = t.parse().expect("unable to parse float value");
                float_val
            })
            .collect::<Vec<Float>>(),
        Err(_) => {
            warn!(@image filename, "error reading file");
            Vec::new()
        }
    }
}

pub fn set_search_directory(options: &mut Options, path: &str) {
    let mut path = std::path::Path::new(path);
    if !path.exists() {
        warn!(@image path.display(), "search directory does not exist");
    }
    if !path.is_dir() {
        path = path.parent().unwrap();
    }

    let path = path.canonicalize().unwrap();
    options.search_directory = Some(path);
}

pub fn resolve_filename(options: &Options, filename: &str) -> String {
    let path = std::path::Path::new(filename);
    if path.is_absolute() || filename.is_empty() || options.search_directory.is_none() {
        return filename.to_string();
    }

    let search_directory = options.search_directory.as_ref().unwrap();
    let mut path = search_directory.clone();
    path.push(filename);
    if path.exists() {
        return path.canonicalize().unwrap().to_str().unwrap().to_string();
    }
    path.to_str().unwrap().to_string()
}
