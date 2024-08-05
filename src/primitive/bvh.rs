use std::{mem::MaybeUninit, sync::{atomic::{AtomicUsize, Ordering}, Arc}};

use crate::{bounds::Union, light::Light, material::Material, reader::paramdict::ParameterDictionary, shape::{AbstractShape, Shape, ShapeIntersection}, Bounds3f, Float, Point3f, Ray, Vec3f};

use super::{Primitive, AbstractPrimitive};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BvhSplitMethod {
    SAH,
    HLBVH,
    Middle,
    EqualCounts,
}

pub struct BvhAggregate {
    max_prims_in_node: usize,
    primitives: Vec<Arc<Primitive>>,
    split_method: BvhSplitMethod,
    nodes: Vec<LinearBvhNode>,
}

impl BvhAggregate {
    pub fn create(
        primitives: Vec<Arc<Primitive>>,
        parameters: &mut ParameterDictionary,
    ) -> BvhAggregate {
        let split_method_name = parameters.get_one_string("splitmethod", "sah");
        let split_method = match split_method_name.as_str() {
            "sah" => BvhSplitMethod::SAH,
            "hlbvh" => BvhSplitMethod::HLBVH,
            "middle" => BvhSplitMethod::Middle,
            "equal" => BvhSplitMethod::EqualCounts,
            _ => panic!("unknown split method {}", split_method_name),
        };

        let max_prims_in_node = parameters.get_one_int("maxnodeprims", 4) as usize;
        BvhAggregate::new(primitives, max_prims_in_node, split_method)
    }

    pub fn new(mut primitives: Vec<Arc<Primitive>>, max_prims_in_node: usize, split_method: BvhSplitMethod) -> BvhAggregate {
        debug_assert!(!primitives.is_empty());

        let mut bvh_primitives: Vec<_> = primitives.iter().enumerate().map(|(i, p)| {
            BvhPrimitive {
                primitive_index: i,
                bounds: p.bounds(),
            }
        }).collect();

        // TODO: Use MaybeUninit
        let mut ordered_primitives: Vec<Option<Arc<Primitive>>> = vec![None; primitives.len()];
        let total_nodes = AtomicUsize::new(0);

        let root = match split_method {
            BvhSplitMethod::HLBVH => {
                Self::build_hlbvh(
                    &mut bvh_primitives,
                    &total_nodes,
                    &mut ordered_primitives,
                )
            },
            _ => {
                let ordered_prims_offset = AtomicUsize::new(0);
                Self::build_recursive(
                    &mut bvh_primitives,
                    &primitives,
                    &total_nodes,
                    &ordered_prims_offset,
                    &mut ordered_primitives,
                    split_method,
                    max_prims_in_node,
                )
            }
        };

        let mut ordered_primitives: Vec<_> = ordered_primitives.into_iter()
            .map(|p| p.expect("not all ordered primitives initialized in BVH construction"))
            .collect();

        std::mem::swap(&mut primitives, &mut ordered_primitives);
        drop(bvh_primitives);

        // TODO: Use MaybeUninit
        let mut nodes: Vec<Option<LinearBvhNode>> = vec![None; total_nodes.load(Ordering::SeqCst)];

        let mut offset = 0;
        Self::flatten_bvh(&mut nodes, Some(Box::new(root)), &mut offset);
        debug_assert_eq!(total_nodes.load(Ordering::SeqCst), offset);

        let nodes: Vec<_> = nodes.into_iter()
            .map(|n| n.expect("not all nodes initialized in flatten_bvh()"))
            .collect();

        BvhAggregate {
            max_prims_in_node: usize::min(255, max_prims_in_node),
            primitives,
            split_method,
            nodes,
        }
    }

    fn build_recursive(
        bvh_primitives: &mut [BvhPrimitive],
        primitives: &Vec<Arc<Primitive>>,
        total_nodes: &AtomicUsize,
        ordered_prims_offset: &AtomicUsize,
        ordered_prims: &mut Vec<Option<Arc<Primitive>>>,
        split_method: BvhSplitMethod,
        max_prims_in_node: usize,
    ) -> BvhBuildNode {
        debug_assert!(!bvh_primitives.is_empty());
        debug_assert!(ordered_prims.len() == primitives.len());

        let mut node = BvhBuildNode::default();

        total_nodes.fetch_add(1, Ordering::SeqCst);
        
        let bounds = bvh_primitives.iter().fold(Bounds3f::default(), |acc, p| -> crate::Bounds3<f32> {
            acc.union(p.bounds)
        });

        if bounds.surface_area() == 0.0 || bvh_primitives.len() == 1 {
            let first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.len(), Ordering::SeqCst);
            for i in 0..bvh_primitives.len() {
                let index = bvh_primitives[i].primitive_index;
                debug_assert!(ordered_prims[first_prim_offset + i].is_none());
                ordered_prims[first_prim_offset + i] = Some(primitives[index].clone());
            }

            node.init_leaf(first_prim_offset, bvh_primitives.len(), bounds);
            return node;
        }

        {
            let centroid_bounds = bvh_primitives.iter().fold(Bounds3f::default(), |acc, p| acc.union(p.centroid()));
            let dim = centroid_bounds.max_dim();
            
            if centroid_bounds.max[dim] == centroid_bounds.min[dim] {
                let first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.len(), Ordering::SeqCst);

                for i in 0..bvh_primitives.len() {
                    let index = bvh_primitives[i].primitive_index;
                    debug_assert!(ordered_prims[first_prim_offset + i].is_none());
                    ordered_prims[first_prim_offset + i] = Some(primitives[index].clone());
                }

                node.init_leaf(first_prim_offset, bvh_primitives.len(), bounds);
                return node;
            } else {
                let split_index = match split_method {
                    BvhSplitMethod::SAH => {
                        if bvh_primitives.len() <= 2 {
                            let split_index = bvh_primitives.len() / 2;
                            pdqselect::select_by(bvh_primitives, split_index, |a, b| {
                                Float::partial_cmp(&a.centroid()[dim], &b.centroid()[dim])
                                    .expect("unexpected NaN")
                            });
                            split_index
                        } else {
                            const N_BUCKETS: usize = 12;
                            let mut buckets = [BvhSplitBucket::default(); N_BUCKETS];

                            for prim in bvh_primitives.iter() {
                                let mut b = (N_BUCKETS as Float * centroid_bounds.offset(prim.centroid())[dim]) as usize;
                                if b == N_BUCKETS { b = N_BUCKETS - 1 };
                                buckets[b].count += 1;
                                buckets[b].bounds = buckets[b].bounds.union(prim.bounds);
                            }

                            const N_SPLITS: usize = N_BUCKETS - 1;
                            let mut costs = [0.0; N_SPLITS];

                            let mut count_below = 0;
                            let mut bounds_below = Bounds3f::default();
                            for i in 0..N_SPLITS {
                                bounds_below = bounds_below.union(buckets[i].bounds);
                                count_below += buckets[i].count;
                                costs[i] += count_below as Float * bounds_below.surface_area();
                            }

                            let mut count_above = 0;
                            let mut bounds_above = Bounds3f::default();
                            for i in (1..=N_SPLITS).rev() {
                                bounds_above = bounds_above.union(buckets[i].bounds);
                                count_above += buckets[i].count;
                                costs[i - 1] += count_above as Float * bounds_above.surface_area();
                            }

                            let mut min_cost_split_bucket = 0;
                            let mut min_cost = Float::INFINITY;
                            for (i, cost) in costs.iter().enumerate() {
                                if *cost < min_cost {
                                    min_cost = *cost;
                                    min_cost_split_bucket = i;
                                }
                            }

                            let leaf_cost = bvh_primitives.len() as Float;
                            min_cost = 0.5 + min_cost / bounds.surface_area();

                            if bvh_primitives.len() > max_prims_in_node || min_cost < leaf_cost {
                                let split_index = itertools::partition(bvh_primitives.iter_mut(), |bp: &BvhPrimitive| {
                                    let mut b = (N_BUCKETS as Float * centroid_bounds.offset(bp.centroid())[dim]) as usize;
                                    if b == N_BUCKETS { b = N_BUCKETS - 1 };
                                    b <= min_cost_split_bucket
                                });

                                split_index
                            } else {
                                let first_prim_offset = ordered_prims_offset.fetch_add(bvh_primitives.len(), Ordering::SeqCst);
                                for i in 0..bvh_primitives.len() {
                                    let index = bvh_primitives[i].primitive_index;
                                    ordered_prims[first_prim_offset + i] = Some(primitives[index].clone());
                                }

                                node.init_leaf(first_prim_offset, bvh_primitives.len(), bounds);
                                return node;
                            }
                        }
                    },
                    BvhSplitMethod::Middle => {
                        let pmid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.0;
                        let split_index = itertools::partition(bvh_primitives.iter_mut(), |pi: &BvhPrimitive| {
                            pi.centroid()[dim] < pmid
                        });

                        if split_index == 0 || split_index == bvh_primitives.len() {
                            let split_index = bvh_primitives.len() / 2;
                            pdqselect::select_by(bvh_primitives, split_index, |a, b| {
                                Float::partial_cmp(&a.centroid()[dim], &b.centroid()[dim])
                                    .expect("unexpected NaN")
                            });
                            split_index
                        } else {
                            split_index
                        }
                    },
                    BvhSplitMethod::EqualCounts => {
                        let split_index = bvh_primitives.len() / 2;
                        pdqselect::select_by(bvh_primitives, split_index, |a, b| {
                            Float::partial_cmp(&a.centroid()[dim], &b.centroid()[dim])
                                .expect("unexpected NaN")
                        });
                        split_index
                    },
                    BvhSplitMethod::HLBVH => unreachable!(),
                };
                
                let left = Box::new(Self::build_recursive(
                    &mut bvh_primitives[0..split_index],
                    primitives,
                    total_nodes,
                    ordered_prims_offset,
                    ordered_prims,
                    split_method,
                    max_prims_in_node,
                ));

                let right = Some(Box::new(Self::build_recursive(
                    &mut bvh_primitives[split_index..],
                    primitives,
                    total_nodes,
                    ordered_prims_offset,
                    ordered_prims,
                    split_method,
                    max_prims_in_node,
                )));

                node.init_interior(dim as u8, left, right)
            }
        }

        node
    }

    fn build_hlbvh(
        bvh_primitives: &mut [BvhPrimitive],
        total_nodes: &AtomicUsize,
        ordered_prims: &mut [Option<Arc<Primitive>>],
    ) -> BvhBuildNode {
        unimplemented!("HLBVH split method not supported")
    }

    fn flatten_bvh(
        linear_nodes: &mut Vec<Option<LinearBvhNode>>,
        node: Option<Box<BvhBuildNode>>,
        offset: &mut usize,
    ) -> usize {
        let node = node.expect("flatten_bvh should be called with a valid root");

        let linear_node_bounds = node.bounds;

        let node_offset = *offset;
        *offset += 1;

        let linear_node = if node.n_primitives > 0 {
            debug_assert!(node.left.is_none() && node.right.is_none());
            debug_assert!(node.n_primitives < 65535);
            let linear_node_n_prims = node.n_primitives as u16;

            LinearBvhNode {
                bounds: linear_node_bounds,
                primitive_offset: node.first_prim_offset,
                second_child_offset: 0,
                n_primitives: linear_node_n_prims,
                axis: 0,
            }
        } else {
            let linear_node_axis = node.split_axis;
            let linear_node_n_prims = 0;
            Self::flatten_bvh(linear_nodes, node.left, offset);
            let second_child_offset = Self::flatten_bvh(linear_nodes, node.right, offset);

            LinearBvhNode {
                bounds: linear_node_bounds,
                primitive_offset: 0,
                second_child_offset,
                n_primitives: linear_node_n_prims,
                axis: linear_node_axis,
            }
        };

        debug_assert!(linear_nodes[node_offset].is_none());
        linear_nodes[node_offset] = Some(linear_node);

        node_offset
    }
}

impl AbstractPrimitive for BvhAggregate {
    fn bounds(&self) -> Bounds3f {
        self.nodes[0].bounds
    }

    fn intersect(&self, ray: &Ray, mut t_max: Float) -> Option<ShapeIntersection> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = Vec3f::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        let dir_is_neg: [u8; 3] = [
            (inv_dir.x < 0.0) as u8,
            (inv_dir.y < 0.0) as u8,
            (inv_dir.z < 0.0) as u8,
        ];

        let mut si: Option<ShapeIntersection> = None;
        let mut to_visit_offset = 0;
        let mut current_node_index = 0;
        let mut nodes_to_visit = [0; 64];

        loop {
            let node = &self.nodes[current_node_index];
            if node.bounds.intersect_p_cached(ray.origin, ray.direction, t_max, inv_dir, dir_is_neg) {
                if node.n_primitives > 0 {
                    for i in 0..node.n_primitives {
                        let prim_si = self.primitives[node.primitive_offset + i as usize].as_ref().intersect(ray, t_max);
                        if let Some(prim_si) = prim_si {
                            t_max = prim_si.t_hit;
                            si = Some(prim_si);
                        }
                    }

                    if to_visit_offset == 0 {
                        break;
                    }

                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                } else if dir_is_neg[node.axis as usize] != 0 {
                    nodes_to_visit[to_visit_offset] = current_node_index + 1;
                    to_visit_offset += 1;
                    current_node_index = node.second_child_offset;
                } else {
                    nodes_to_visit[to_visit_offset] = node.second_child_offset;
                    to_visit_offset += 1;
                    current_node_index += 1;
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }

                to_visit_offset -= 1;
                current_node_index = nodes_to_visit[to_visit_offset];
            }
        }

        si
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let inv_dir = Vec3f::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        let dir_is_neg: [u8; 3] = [
            (inv_dir.x < 0.0) as u8,
            (inv_dir.y < 0.0) as u8,
            (inv_dir.z < 0.0) as u8,
        ];

        let mut to_visit_offset = 0;
        let mut current_node_index = 0;
        let mut nodes_to_visit = [0; 64];

        loop {
            let node = &self.nodes[current_node_index];

            if node.bounds.intersect_p_cached(ray.origin, ray.direction, t_max, inv_dir, dir_is_neg) {
                if node.n_primitives > 0 {
                    for i in 0..node.n_primitives {
                        if self.primitives[node.primitive_offset + i as usize].as_ref().intersect_predicate(ray, t_max) {
                            return true;
                        }
                    }

                    if to_visit_offset == 0 {
                        break;
                    }

                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                } else if dir_is_neg[node.axis as usize] != 0 {
                    nodes_to_visit[to_visit_offset] = current_node_index + 1;
                    to_visit_offset += 1;
                    current_node_index = node.second_child_offset;
                } else {
                    nodes_to_visit[to_visit_offset] = node.second_child_offset;
                    to_visit_offset += 1;
                    current_node_index += 1;
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }

                to_visit_offset -= 1;
                current_node_index = nodes_to_visit[to_visit_offset];
            }
        }

        false
    }
}

fn left_shift_3(mut x: u32) -> u32 {
    if x == (1 << 10) { x -= 1 };
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    (x | (x << 2)) & 0b00001001001001001001001001001001
}

fn encode_morton_3(x: Float, y: Float, z: Float) -> u32 {
    (left_shift_3(z as u32) << 2) | (left_shift_3(y as u32) << 1) | left_shift_3(x as u32)
}

#[repr(align(32))]
#[derive(Debug, Copy, Clone)]
struct LinearBvhNode {
    bounds: Bounds3f,
    primitive_offset: usize,
    second_child_offset: usize,
    n_primitives: u16,
    axis: u8
}

#[derive(Default, Clone)]
struct BvhBuildNode {
    bounds: Bounds3f,
    left: Option<Box<BvhBuildNode>>,
    right: Option<Box<BvhBuildNode>>,
    split_axis: u8,
    first_prim_offset: usize,
    n_primitives: usize,
}

impl BvhBuildNode {
    pub fn init_leaf(&mut self, first_prim_offset: usize, n_primitives: usize, bounds: Bounds3f) {
        self.bounds = bounds;
        self.left = None;
        self.right = None;
        self.split_axis = 0;
        self.first_prim_offset = first_prim_offset;
        self.n_primitives = n_primitives;
    }

    pub fn init_interior(&mut self, split_axis: u8, left: Box<BvhBuildNode>, right: Option<Box<BvhBuildNode>>) {
        let bounds = if let Some(ref right) = right {
            left.bounds.union(right.bounds)
        } else {
            left.bounds
        };

        self.bounds = bounds;
        self.left = Some(left);
        self.right = right;
        self.split_axis = split_axis;
        self.first_prim_offset = 0;
        self.n_primitives = 0;
    }
}

struct BvhPrimitive {
    primitive_index: usize,
    bounds: Bounds3f,
}

impl BvhPrimitive {
    pub fn centroid(&self) -> Point3f {
        0.5 * self.bounds.min + self.bounds.max * 0.5
    }
}

#[derive(Default, Clone, Copy)]
struct BvhSplitBucket {
    count: i32,
    bounds: Bounds3f,
}

#[derive(Default, Clone, Copy)]
struct MortonPrimitive {
    primitive_index: usize,
    morton_code: u32,
}

struct LBvhTreelet {
    start_index: usize,
    n_primitives: usize,
    build_nodes: Vec<BvhBuildNode>,
}
