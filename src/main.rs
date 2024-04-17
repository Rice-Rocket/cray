use crate::math::*;

pub mod math;

fn main() {
    let m = Mat4::from_euler_angles(math::math::FRAC_PI_2, 0.0, 0.0);
    println!("{}", m);
}
