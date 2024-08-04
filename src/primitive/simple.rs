use std::sync::Arc;

use crate::{material::Material, shape::{Shape, ShapeIntersection, AbstractShape}, Bounds3f, Float, Ray};

use super::AbstractPrimitive;

pub struct SimplePrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
}

impl AbstractPrimitive for SimplePrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray.clone(), t_max)?;
        si.intr.set_intersection_properties(&self.material, &None, &None, &ray.medium);
        Some(si)
    }

    fn intersect_predicate(&self, ray: Ray, t_max: Float) -> bool {
        self.shape.intersect_predicate(ray, t_max)
    }
}
