use std::sync::Arc;

use crate::{bounds::Union, reader::paramdict::ParameterDictionary, shape::ShapeIntersection, Bounds3f, Float, Ray};

use super::{AbstractPrimitive, Primitive};

#[derive(Debug, Clone)]
pub struct BasicListAggregate {
    primitives: Vec<Arc<Primitive>>,
}

impl BasicListAggregate {
    pub fn create(
        primitives: Vec<Arc<Primitive>>,
        parameters: &mut ParameterDictionary,
    ) -> BasicListAggregate {
        BasicListAggregate { primitives }
    }

    pub fn new(
        primitives: Vec<Arc<Primitive>>,
    ) -> BasicListAggregate {
        BasicListAggregate { primitives }
    }
}

impl AbstractPrimitive for BasicListAggregate {
    fn bounds(&self) -> Bounds3f {
        let mut bounds = Bounds3f::default();
        for prim in self.primitives.iter() {
            bounds = bounds.union(prim.bounds());
        }
        bounds
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut t_closest = Float::MAX;
        let mut isect = None;

        for prim in self.primitives.iter() {
            if let Some(i) = prim.intersect(ray, t_max) {
                if i.t_hit < t_closest {
                    t_closest = i.t_hit;
                    isect = Some(i);
                }
            }
        }

        isect
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        for prim in self.primitives.iter() {
            if prim.intersect_predicate(ray, t_max) {
                return true;
            }
        }

        false
    }
}
