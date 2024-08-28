use std::sync::Arc;

use crate::{light::Light, material::Material, media::MediumInterface, shape::{AbstractShape, Shape, ShapeIntersection}, Bounds3f, Float, Ray};

use super::AbstractPrimitive;

#[derive(Debug, Clone)]
pub struct GeometricPrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
    pub area_light: Option<Arc<Light>>,
    pub medium_interface: Arc<MediumInterface>,
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<Shape>,
        material: Arc<Material>,
        area_light: Option<Arc<Light>>,
        medium_interface: Arc<MediumInterface>,
    ) -> GeometricPrimitive {
        GeometricPrimitive {
            shape,
            material,
            area_light,
            medium_interface,
        }
    }
}

impl AbstractPrimitive for GeometricPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray, t_max)?;
        debug_assert!(si.t_hit < 1.001 * t_max);

        si.intr.set_intersection_properties(&self.material, self.area_light.as_ref(), Some(&self.medium_interface), ray.medium.as_ref());
        Some(si)
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        self.intersect(ray, t_max).is_some()
    }
}
