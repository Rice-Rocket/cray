pub mod bounds2;
pub mod bounds3;
pub mod direction;

pub use bounds2::Bounds2;
pub use bounds3::Bounds3;

use crate::math::Float;


/// A 3-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds3f = Bounds3<Float>;
/// A 2-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds2f = Bounds2<Float>;
/// A 3-dimensional axis-aligned bounding box of type [`i32`]
pub type Bounds3i = Bounds3<i32>;
/// A 2-dimensional axis-aligned bounding box of type [`i32`]
pub type Bounds2i = Bounds2<i32>;
/// A 3-dimensional axis-aligned bounding box of type [`u32`]
pub type Bounds3u = Bounds3<u32>;
/// A 2-dimensional axis-aligned bounding box of type [`u32`]
pub type Bounds2u = Bounds2<u32>;


pub trait Union<T> {
    type Output;

    fn union(self, rhs: T) -> Self::Output;
}
