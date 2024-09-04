use std::{collections::HashMap, mem::MaybeUninit, sync::Arc};

use crate::{bounds::Union, lerp, light::{AbstractLight, Light, LightBounds, LightSampleContext}, safe, sampling::sample_discrete, sqr, Bounds3f, DirectionCone, Dot, Float, Normal3f, OctahedralVec3, Point3f, ONE_MINUS_EPSILON, PI};

use super::{AbstractLightSampler, SampledLight};

pub struct BvhLightSampler {
    lights: Arc<[Arc<Light>]>,
    infinite_lights: Arc<[Arc<Light>]>,
    all_light_bounds: Bounds3f,
    nodes: Vec<LightBvhNode>,
    light_to_bit_trail: HashMap<usize, u32>,
}

impl BvhLightSampler {
    pub fn new(lights: Arc<[Arc<Light>]>) -> BvhLightSampler {
        let mut bvh_lights = Vec::new();
        let mut infinite_lights = Vec::new();
        let mut all_light_bounds = Bounds3f::default();

        for i in 0..lights.len() {
            let light = &lights[i];
            let Some(ref light_bounds) = light.bounds() else {
                infinite_lights.push(light.clone());
                continue;
            };

            if light_bounds.phi > 0.0 {
                bvh_lights.push((i as u32, *light_bounds));
                all_light_bounds = all_light_bounds.union(light_bounds.bounds);
            }
        }

        let mut light_sampler = BvhLightSampler {
            lights,
            infinite_lights: infinite_lights.into(),
            all_light_bounds,
            nodes: Vec::new(),
            light_to_bit_trail: HashMap::new(),
        };

        if !bvh_lights.is_empty() {
            let n_bvh_lights = bvh_lights.len();
            light_sampler.build_bvh(&mut bvh_lights, 0, n_bvh_lights, 0, 0);
        }

        light_sampler
    }
    
    fn build_bvh(
        &mut self,
        bvh_lights: &mut [(u32, LightBounds)],
        start: usize,
        end: usize,
        bit_trail: u32,
        depth: i32,
    ) -> (u32, LightBounds) {
        if end - start == 1 {
            let node_index = self.nodes.len();
            let cb = CompactLightBounds::new(&bvh_lights[start].1, &self.all_light_bounds);
            let light_index = bvh_lights[start].0;
            self.nodes.push(LightBvhNode::make_leaf(light_index, cb));
            self.light_to_bit_trail.insert((self.lights[light_index as usize].as_ref() as *const Light) as usize, bit_trail);

            return (node_index as u32, bvh_lights[start].1)
        }

        let mut bounds = Bounds3f::default();
        let mut centroid_bounds = Bounds3f::default();
        for (_, lb) in bvh_lights.iter().take(end).skip(start) {
            bounds = bounds.union(lb.bounds);
            centroid_bounds = centroid_bounds.union(lb.centroid());
        }

        let mut min_cost = Float::INFINITY;
        let mut min_cost_split_bucket = -1;
        let mut min_cost_split_dim = 0;
        const N_BUCKETS: i32 = 12;

        for dim in 0..3 {
            if centroid_bounds.max[dim] == centroid_bounds.min[dim] {
                continue;
            }

            let mut bucket_light_bounds = [LightBounds::default(); N_BUCKETS as usize];
            for (_, lb) in bvh_lights.iter().take(end).skip(start) {
                let pc = lb.centroid();
                let mut b = (N_BUCKETS as Float * centroid_bounds.offset(pc)[dim]) as i32;
                if b == N_BUCKETS {
                    b = N_BUCKETS - 1;
                }
                debug_assert!(b >= 0);
                debug_assert!(b < N_BUCKETS);

                bucket_light_bounds[b as usize] = bucket_light_bounds[b as usize].union(lb);
            }

            let mut costs = [0.0; (N_BUCKETS as usize) - 1];
            for (i, cost) in costs.iter_mut().enumerate() {
                let mut b0 = LightBounds::default();
                let mut b1 = LightBounds::default();

                for lb in bucket_light_bounds.iter().take(i + 1) {
                    b0 = b0.union(lb);
                }

                for lb in bucket_light_bounds.iter().skip(i + 1) {
                    b1 = b1.union(lb);
                }

                *cost = Self::evaluate_cost(&b0, &bounds, dim as i32) + Self::evaluate_cost(&b1, &bounds, dim as i32);
            }

            for (i, cost) in costs.iter().enumerate().skip(1) {
                if *cost > 0.0 && *cost < min_cost {
                    min_cost = *cost;
                    min_cost_split_bucket = i as i32;
                    min_cost_split_dim = dim;
                }
            }
        }

        let mid = if min_cost_split_bucket == -1 {
            (start + end) / 2
        } else {
            let mut mid = itertools::partition(bvh_lights[start..end].iter_mut(), |l: &(u32, LightBounds)| {
                let mut b = (N_BUCKETS as Float * centroid_bounds.offset(l.1.centroid())[min_cost_split_dim]) as i32;
                if b == N_BUCKETS {
                    b = N_BUCKETS - 1;
                }

                debug_assert!(b >= 0);
                debug_assert!(b < N_BUCKETS);
                b <= min_cost_split_bucket
            });

            mid += start;
            
            if mid == start || mid == end {
                mid = (start + end) / 2;
            }

            debug_assert!(mid > start && mid < end);

            mid
        };

        let node_index = self.nodes.len();
        self.nodes.push(LightBvhNode::default());
        debug_assert!(depth < 64);
        let child0 = self.build_bvh(bvh_lights, start, mid, bit_trail, depth + 1);
        debug_assert_eq!(node_index + 1, child0.0 as usize);
        let child1 = self.build_bvh(bvh_lights, mid, end, bit_trail | (1 << depth), depth + 1);

        let lb = child0.1.union(&child1.1);
        let cb = CompactLightBounds::new(&lb, &self.all_light_bounds);
        self.nodes[node_index] = LightBvhNode::make_interior(child1.0, cb);

        (node_index as u32, lb)
    }

    fn evaluate_cost(b: &LightBounds, bounds: &Bounds3f, dim: i32) -> Float {
        let theta_o = b.cos_theta_o.acos();
        let theta_e = b.cos_theta_e.acos();
        let theta_w = Float::min(theta_o + theta_e, PI);
        let sin_theta_o = safe::sqrt(1.0 - sqr(b.cos_theta_o));
        let m_omega = 2.0 * PI * (1.0 - b.cos_theta_o)
            + PI / 2.0
            * (2.0 * theta_w * sin_theta_o - Float::cos(theta_o - 2.0 * theta_w)
            - 2.0 * theta_o * sin_theta_o + b.cos_theta_o);

        let kr = bounds.diagonal().max_element() / bounds.diagonal()[dim as usize];
        b.phi * m_omega * kr * b.bounds.surface_area()
    }
}

impl AbstractLightSampler for BvhLightSampler {
    fn sample(&self, ctx: &LightSampleContext, mut u: Float) -> Option<SampledLight> {
        let p_infinite = self.infinite_lights.len() as Float 
            / (self.infinite_lights.len() + (if self.nodes.is_empty() { 0 } else { 1 })) as Float;

        if u < p_infinite {
            u /= p_infinite;
            let index = ((u * self.infinite_lights.len() as Float) as usize).min(self.infinite_lights.len() - 1);
            let pmf = p_infinite / self.infinite_lights.len() as Float;
            Some(SampledLight { light: self.infinite_lights[index].clone(), p: pmf })
        } else {
            if self.nodes.is_empty() {
                return None;
            }

            let p = ctx.p();
            let n = ctx.ns;
            u = Float::min((u - p_infinite) / (1.0 - p_infinite), ONE_MINUS_EPSILON);
            let mut node_index = 0;
            let mut pmf = 1.0 - p_infinite;

            loop {
                let node = &self.nodes[node_index];
                if !node.is_leaf {
                    let children = (
                        &self.nodes[node_index + 1],
                        &self.nodes[node.child_or_light_index as usize],
                    );

                    let ci = [
                        children.0.light_bounds.importance(p, n, &self.all_light_bounds),
                        children.1.light_bounds.importance(p, n, &self.all_light_bounds),
                    ];

                    if ci[0] == 0.0 && ci[1] == 0.0 {
                        return None;
                    }

                    let mut node_pmf = 0.0;
                    let child = sample_discrete(&ci, u, Some(&mut node_pmf), Some(&mut u));
                    pmf *= node_pmf;
                    node_index = if child == Some(0) { node_index + 1 } else { node.child_or_light_index as usize };
                } else {
                    if node_index > 0 {
                        debug_assert!(node.light_bounds.importance(p, n, &self.all_light_bounds) > 0.0);
                    }

                    if node_index > 0 || node.light_bounds.importance(p, n, &self.all_light_bounds) > 0.0 {
                        return Some(SampledLight { light: self.lights[node.child_or_light_index as usize].clone(), p: pmf });
                    }

                    return None;
                }
            }
        }
    }

    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float {
        match self.light_to_bit_trail.get(&((light as *const Light) as usize)) {
            Some(bit_trail) => {
                let mut bit_trail = *bit_trail;

                let p = ctx.p();
                let n = ctx.ns;
                let p_infinite = self.infinite_lights.len() as Float
                    / (self.infinite_lights.len() + if self.nodes.is_empty() { 0 } else { 1 }) as Float;

                let mut pmf = 1.0 - p_infinite;
                let mut node_index = 0;

                loop {
                    let node = &self.nodes[node_index];
                    if node.is_leaf {
                        debug_assert!(std::ptr::eq(light, self.lights[node.child_or_light_index as usize].as_ref()));
                        return pmf;
                    }

                    let child0 = &self.nodes[node_index + 1];
                    let child1 = &self.nodes[node.child_or_light_index as usize];
                    let ci = [
                        child0.light_bounds.importance(p, n, &self.all_light_bounds),
                        child1.light_bounds.importance(p, n, &self.all_light_bounds),
                    ];
                    debug_assert!(ci[(bit_trail & 1) as usize] > 0.0);
                    pmf *= ci[(bit_trail & 1) as usize] / (ci[0] + ci[1]);

                    node_index = if bit_trail & 1 != 0 { node.child_or_light_index as usize } else { node_index + 1 };
                    bit_trail >>= 1;
                }
            },
            None => 1.0 / (self.infinite_lights.len() + if self.nodes.is_empty() { 0 } else { 1 }) as Float,
        }
    }

    fn sample_light(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }

        let light_index = ((u * self.lights.len() as Float) as usize).min(self.lights.len() - 1);
        Some(SampledLight { light: self.lights[light_index].clone(), p: 1.0 / self.lights.len() as Float })
    }

    fn pmf_light(&self, light: &Light) -> Float {
        if self.lights.is_empty() {
            return 0.0;
        }

        1.0 / self.lights.len() as Float
    }
}

#[repr(packed(2))]
pub struct CompactLightBounds {
    w: OctahedralVec3,
    phi: Float,
    q_cos_theta_o: u16,
    q_cos_theta_e: u16,
    two_sided: bool,
    qb: [[u16; 3]; 2],
}

impl CompactLightBounds {
    pub fn new(lb: &LightBounds, allb: &Bounds3f) -> CompactLightBounds {
        let mut qb = [[0; 3]; 2];

        for c in 0..3 {
            qb[0][c] = Float::floor(Self::quantize_bounds(lb.bounds[0][c], allb.min[c], allb.max[c])) as u16;
            qb[1][c] = Float::ceil(Self::quantize_bounds(lb.bounds[1][c], allb.min[c], allb.max[c])) as u16;
        }

        CompactLightBounds {
            w: OctahedralVec3::new(lb.w.normalize()),
            phi: lb.phi,
            q_cos_theta_o: Self::quantize_cos(lb.cos_theta_o),
            q_cos_theta_e: Self::quantize_cos(lb.cos_theta_e),
            two_sided: lb.two_sided,
            qb,
        }
    }

    #[inline]
    pub fn two_sided(&self) -> bool {
        self.two_sided
    }

    #[inline]
    pub fn cos_theta_o(&self) -> Float {
        2.0 * (self.q_cos_theta_o as Float / 32767.0) - 1.0
    }

    #[inline]
    pub fn cos_theta_e(&self) -> Float {
        2.0 * (self.q_cos_theta_e as Float / 32767.0) - 1.0
    }

    #[inline]
    pub fn bounds(&self, allb: &Bounds3f) -> Bounds3f {
        Bounds3f::new(
            Point3f::new(
                lerp(allb.min.x, allb.max.x, self.qb[0][0] as Float / 65535.0),
                lerp(allb.min.y, allb.max.y, self.qb[0][1] as Float / 65535.0),
                lerp(allb.min.z, allb.max.z, self.qb[0][2] as Float / 65535.0),
            ),
            Point3f::new(
                lerp(allb.min.x, allb.max.x, self.qb[1][0] as Float / 65535.0),
                lerp(allb.min.y, allb.max.y, self.qb[1][1] as Float / 65535.0),
                lerp(allb.min.z, allb.max.z, self.qb[1][2] as Float / 65535.0),
            ),
        )
    }

    #[inline]
    fn quantize_cos(c: Float) -> u16 {
        debug_assert!((-1.0..=1.0).contains(&c));
        Float::floor(32767.0 * ((c + 1.0) / 2.0)) as u16
    }

    #[inline]
    fn quantize_bounds(c: Float, min: Float, max: Float) -> Float {
        debug_assert!(c >= min && c <= max);
        if min == max {
            0.0
        } else {
            65535.0 * Float::clamp((c - min) / (max - min), 0.0, 1.0)
        } 
    }

    pub fn importance(&self, p: Point3f, n: Normal3f, allb: &Bounds3f) -> Float {
        let cos_theta_o = self.cos_theta_o();
        let cos_theta_e = self.cos_theta_e();

        let pc = (allb.min + allb.max) / 2.0;
        let mut d2 = p.distance_squared(pc);
        d2 = Float::max(d2, allb.diagonal().length() / 2.0);

        let cos_sub_clamped = |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| -> Float {
            if cos_theta_a > cos_theta_b {
                return 1.0;
            }

            cos_theta_a * cos_theta_b + sin_theta_a * sin_theta_b
        };

        let sin_sub_clamped = |sin_theta_a: Float, cos_theta_a: Float, sin_theta_b: Float, cos_theta_b: Float| -> Float {
            if cos_theta_a > cos_theta_b {
                return 0.0;
            }

            sin_theta_a * cos_theta_b - cos_theta_a * sin_theta_b
        };

        let wi = (p - pc).normalize();
        let mut cos_theta_w = self.w.to_vec3().dot(wi);
        if self.two_sided {
            cos_theta_w = cos_theta_w.abs();
        }
        let sin_theta_w = safe::sqrt(1.0 - sqr(cos_theta_w));

        let cos_theta_b = DirectionCone::bound_subtended_directions(*allb, p).cos_theta;
        let sin_theta_b = safe::sqrt(1.0 - sqr(cos_theta_b));

        let sin_theta_o = safe::sqrt(1.0 - sqr(cos_theta_o));
        let cos_theta_x = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, cos_theta_o);
        let sin_theta_x = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, cos_theta_o);
        let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);

        if cos_theta_p <= cos_theta_e {
            return 0.0;
        }

        let mut importance = self.phi * cos_theta_p / d2;
        debug_assert!(importance >= -1e-3);

        if n != Normal3f::ZERO {
            let cos_theta_i = wi.dot(n).abs();
            let sin_theta_i = safe::sqrt(1.0 - sqr(cos_theta_i));
            let cos_theta_pi = cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
            importance *= cos_theta_pi;
        }
        
        importance = Float::max(importance, 0.0);
        importance
    }
}

impl Default for CompactLightBounds {
    fn default() -> Self {
        Self {
            w: OctahedralVec3::default(),
            phi: 0.0,
            q_cos_theta_o: 0,
            q_cos_theta_e: 0,
            two_sided: false,
            qb: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct LightBvhNode {
    pub light_bounds: CompactLightBounds,
    pub child_or_light_index: u32,
    pub is_leaf: bool,
}

impl LightBvhNode {
    pub fn make_leaf(light_index: u32, cb: CompactLightBounds) -> LightBvhNode {
        LightBvhNode {
            light_bounds: cb,
            child_or_light_index: light_index,
            is_leaf: true,
        }
    }

    pub fn make_interior(child_1_index: u32, cb: CompactLightBounds) -> LightBvhNode {
        LightBvhNode {
            light_bounds: cb,
            child_or_light_index: child_1_index,
            is_leaf: false,
        }
    }
}
