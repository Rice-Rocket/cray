pub mod bounds2;
pub mod bounds3;
pub mod direction;

pub use bounds2::TBounds2;
pub use bounds3::TBounds3;

use crate::math::Scalar;


/// A 3-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds3 = TBounds3<Scalar>;
/// A 2-dimensional axis-aligned bounding box of type [`Scalar`]
pub type Bounds2 = TBounds2<Scalar>;
/// A 3-dimensional axis-aligned bounding box of type [`i32`]
pub type IBounds3 = TBounds3<i32>;
/// A 2-dimensional axis-aligned bounding box of type [`i32`]
pub type IBounds2 = TBounds2<i32>;
/// A 3-dimensional axis-aligned bounding box of type [`u32`]
pub type UBounds3 = TBounds3<u32>;
/// A 2-dimensional axis-aligned bounding box of type [`u32`]
pub type UBounds2 = TBounds2<u32>;
