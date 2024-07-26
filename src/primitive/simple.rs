use std::sync::Arc;

use crate::{material::Material, shape::{Shape, ShapeIntersection, ShapeLike}, Bounds3f, Float, Ray};

use super::PrimitiveLike;

pub struct SimplePrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
}

impl PrimitiveLike for SimplePrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray, t_max)?;
        si.intr.set_intersection_properties(&self.material, &None);
        Some(si)
    }

    fn intersect_predicate(&self, ray: Ray, t_max: Float) -> bool {
        self.shape.intersect_predicate(ray, t_max)
    }
}
