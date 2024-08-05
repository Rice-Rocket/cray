use bvh::BvhAggregate;
use geometric::GeometricPrimitive;
use simple::SimplePrimitive;
use transformed::TransformedPrimitive;

use crate::{shape::ShapeIntersection, Bounds3f, Float, Ray};

pub mod simple;
pub mod transformed;
pub mod geometric;
pub mod bvh;

pub trait AbstractPrimitive {
    /// The bounding box of the primitive
    fn bounds(&self) -> Bounds3f;

    /// Intersects a ray with the primitive, returning [`None`] if there was no intersection. 
    /// Otherwise, returns a [`ShapeIntersection`] with information about the intersection.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Determines *if* the ray would intersect with the primitive.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;
}

pub enum Primitive {
    Simple(SimplePrimitive),
    Transformed(TransformedPrimitive),
    BvhAggregate(BvhAggregate),
    Geometric(GeometricPrimitive),
}

impl AbstractPrimitive for Primitive {
    fn bounds(&self) -> Bounds3f {
        match self {
            Primitive::Simple(p) => p.bounds(),
            Primitive::Transformed(p) => p.bounds(),
            Primitive::BvhAggregate(p) => p.bounds(),
            Primitive::Geometric(p) => p.bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Primitive::Simple(p) => p.intersect(ray, t_max),
            Primitive::Transformed(p) => p.intersect(ray, t_max),
            Primitive::BvhAggregate(p) => p.intersect(ray, t_max),
            Primitive::Geometric(p) => p.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Primitive::Simple(p) => p.intersect_predicate(ray, t_max),
            Primitive::Transformed(p) => p.intersect_predicate(ray, t_max),
            Primitive::BvhAggregate(p) => p.intersect_predicate(ray, t_max),
            Primitive::Geometric(p) => p.intersect_predicate(ray, t_max),
        }
    }
}
