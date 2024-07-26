use std::sync::Arc;

use crate::{light::Light, material::Material, shape::{Shape, ShapeIntersection, ShapeLike}, Bounds3f, Float, Ray};

use super::PrimitiveLike;

pub struct GeometricPrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
    pub area_light: Option<Arc<Light>>,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<Shape>,
        material: Arc<Material>,
        area_light: Option<Arc<Light>>,
    ) -> GeometricPrimitive {
        GeometricPrimitive {
            shape,
            material,
            area_light
        }
    }
}

impl PrimitiveLike for GeometricPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray, t_max)?;
        debug_assert!(si.t_hit < 1.001 * t_max);

        si.intr.set_intersection_properties(&self.material, &self.area_light);
        Some(si)
    }

    fn intersect_predicate(&self, ray: Ray, t_max: Float) -> bool {
        self.intersect(ray, t_max).is_some()
    }
}
