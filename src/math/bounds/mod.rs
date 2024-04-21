pub mod bounds2;
pub mod bounds3;
pub mod direction;

pub use bounds2::TBounds2;
pub use bounds3::TBounds3;

use crate::math::Scalar;


/// A 3-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds3f = TBounds3<Scalar>;
/// A 2-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds2f = TBounds2<Scalar>;
/// A 3-dimensional axis-aligned bounding box of type [`i32`]
pub type Bounds3i = TBounds3<i32>;
/// A 2-dimensional axis-aligned bounding box of type [`i32`]
pub type Bounds2i = TBounds2<i32>;
/// A 3-dimensional axis-aligned bounding box of type [`u32`]
pub type Bounds3u = TBounds3<u32>;
/// A 2-dimensional axis-aligned bounding box of type [`u32`]
pub type Bounds2u = TBounds2<u32>;
