#![allow(dead_code)]

pub mod ray;

#[allow(unused_imports)]
pub use nalgebra::{self as na, Transform3, Unit};
#[allow(unused_imports)]
pub use nalgebra_glm::{self as glm, vec2, vec3, vec4};
use nalgebra_glm::{Qua, TMat2, TMat3, TMat4, TVec2, TVec3, TVec4};
pub use ray::Ray;


pub type Scalar = f32;

pub type Quat = Qua<Scalar>;
pub type Mat2 = TMat2<Scalar>;
pub type Mat3 = TMat3<Scalar>;
pub type Mat4 = TMat4<Scalar>;
pub type Vec2 = TVec2<Scalar>;
pub type Vec3 = TVec3<Scalar>;
pub type Vec4 = TVec4<Scalar>;
