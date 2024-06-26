use std::fs;

use tracing::warn;

use crate::Scalar;

pub fn read_float_file(filename: &str) -> Vec<Scalar> {
    let contents = fs::read_to_string(filename);
    match contents {
        Ok(s) => s
            .split_whitespace()
            .map(|t| {
                let float_val: Scalar = t.parse().expect("unable to parse float value");
                float_val
            })
            .collect::<Vec<Scalar>>(),
        Err(_) => {
            warn!("error reading file {}", filename);
            Vec::new()
        }
    }
}
