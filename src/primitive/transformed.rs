use std::sync::Arc;

use crate::{light::Light, material::Material, shape::{Shape, ShapeIntersection, ShapeLike}, transform::Transform, Bounds3f, Float, Ray};

use super::{Primitive, PrimitiveLike};

pub struct TransformedPrimitive {
    primitive: Arc<Primitive>,
    render_from_primitive: Transform,
}

impl TransformedPrimitive {
    pub fn new(
        primitive: Arc<Primitive>,
        render_from_primitive: Transform
    ) -> TransformedPrimitive {
        TransformedPrimitive {
            primitive,
            render_from_primitive,
        }
    }
}

impl PrimitiveLike for TransformedPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.render_from_primitive * self.primitive.bounds()
    }

    fn intersect(&self, ray: Ray, mut t_max: Float) -> Option<ShapeIntersection> {
        let ray = self.render_from_primitive.inverse().mul_ray(ray, Some(&mut t_max));
        let mut si = self.primitive.intersect(ray, t_max)?;
        debug_assert!(si.t_hit <= 1.001 * t_max);

        si.intr = self.render_from_primitive * si.intr;
        debug_assert!(si.intr.interaction.n.dot(si.intr.shading.n) >= 0.0);

        Some(si)
    }

    fn intersect_predicate(&self, ray: Ray, mut t_max: Float) -> bool {
        let ray = self.render_from_primitive.mul_ray(ray, Some(&mut t_max));
        self.primitive.intersect_predicate(ray, t_max)
    }
}