use std::sync::Arc;

use crate::{bounds::Union, error, gamma, interaction::{Interaction, SurfaceInteraction}, lerp, lerp_float, numeric::DifferenceOfProducts, quadratic, reader::{error::ParseResult, paramdict::ParameterDictionary, target::FileLoc}, sampling::{bilinear_pdf, invert_bilinear, invert_spherical_rectangle_sample, sample_bilinear, sample_spherical_rectangle}, spherical_quad_area, transform::{ApplyTransform, Transform}, Bounds3f, DirectionCone, Dot, Float, Mat3, Normal3f, Point2f, Point3f, Point3fi, Ray, Vec3f};

use super::{mesh::BilinearPatchMesh, AbstractShape, Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

#[derive(Debug, Clone)]
pub struct BilinearPatch {
    mesh: Arc<BilinearPatchMesh>,
    blp_index: usize,
    area: Float,
}

pub struct BilinearIntersection {
    pub uv: Point2f,
    pub t: Float,
}

impl BilinearPatch {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 1e-4;
    
    pub fn create_mesh(
        render_from_object: &Transform,
        reverse_orientation: bool,
        parameters: &mut ParameterDictionary,
        loc: &FileLoc,
    ) -> ParseResult<BilinearPatchMesh> {
        let mut vi = parameters.get_int_array("indices")?;
        let p = parameters.get_point3f_array("P")?;
        let uvs = parameters.get_point2f_array("uv")?;

        if vi.is_empty() {
            if p.len() == 4 {
                vi = vec![0, 1, 2, 3];
            } else {
                error!(loc, "vertex indices 'indices' not provided with bilinear patch mesh shape");
            }
        } else if vi.len() % 4 != 0 {
            error!(loc, "number of vertex indices 'indices' not multiple of 4 as expected");
            // TODO: Could just pop excess and warn
        }

        if p.is_empty() {
            error!(loc, "vertex positions 'P' not provided with bilinear path mesh shape");
        }

        if !uvs.is_empty() && uvs.len() != p.len() {
            error!(loc, "number of vertex positions 'P' and vertex UVs 'uv' do not match");
            // TODO: Could just dicard uvs instead of panicing + warn
        }

        let n = parameters.get_normal3f_array("N")?;
        if !n.is_empty() && n.len() != p.len() {
            error!(loc, "number of vertex positions 'P' and vertex normals 'N' do not match");
            // TODO: Could just discard instead of pancing + warn
        }

        for v in vi.iter() {
            if *v as usize >= p.len() {
                error!(loc, "vertex indices {} out of bounds 'P' array length {}", v, p.len());
            }
        }

        let face_indices = parameters.get_int_array("faceIndices")?;
        if !face_indices.is_empty() && face_indices.len() != vi.len() / 4 {
            error!(loc, "number of face indices 'faceIndices' and vertex indices 'indices' do not match");
            // TODO: Could just discard instead of pancing + warn
        }

        // TODO: Handle image distributions

        Ok(BilinearPatchMesh::new(
            render_from_object,
            reverse_orientation,
            vi.into_iter().map(|i| i as usize).collect(),
            p,
            n,
            uvs,
            face_indices.into_iter().map(|i| i as usize).collect(),
        ))
    }

    pub fn new(mesh: Arc<BilinearPatchMesh>, blp_index: usize) -> BilinearPatch {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&mesh, blp_index);
        let area = if BilinearPatch::is_rectangle(&mesh, blp_index) {
            p00.distance(p01) * p00.distance(p10)
        } else {
            const NA: usize = 3;
            let mut p = [[Point3f::ZERO; NA + 1]; NA + 1];
            for (i, pi) in p.iter_mut().enumerate() {
                let u = i as Float / NA as Float;
                for (j, pij) in pi.iter_mut().enumerate() {
                    let v = j as Float / NA as Float;
                    *pij = lerp_float(lerp_float(p00, p01, v), lerp_float(p10, p11, v), u);
                }
            }

            let mut area = 0.0;
            for i in 0..NA {
                for j in 0..NA {
                    area += 0.5 * (p[i + 1][j + 1] - p[i][j]).cross(p[i + 1][j] - p[i][j + 1]).length();
                }
            }

            area
        };

        BilinearPatch {
            mesh,
            blp_index,
            area,
        }
    }

    pub fn create_patches(mesh: Arc<BilinearPatchMesh>) -> Vec<Arc<Shape>> {
        let mut patches = Vec::new();
        for i in 0..mesh.n_patches {
            patches.push(Arc::new(Shape::BilinearPatch(Box::new(BilinearPatch::new(mesh.clone(), i)))));
        }
        patches
    }

    fn get_points(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> (Point3f, Point3f, Point3f, Point3f) {
        let p00 = mesh.p[mesh.vertex_indices[blp_index * 4]];
        let p10 = mesh.p[mesh.vertex_indices[blp_index * 4 + 1]];
        let p01 = mesh.p[mesh.vertex_indices[blp_index * 4 + 2]];
        let p11 = mesh.p[mesh.vertex_indices[blp_index * 4 + 3]];
        (p00, p10, p01, p11)
    }

    fn get_vertex_indices(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> (usize, usize, usize, usize) {
        (
            mesh.vertex_indices[blp_index * 4],
            mesh.vertex_indices[blp_index * 4 + 1],
            mesh.vertex_indices[blp_index * 4 + 2],
            mesh.vertex_indices[blp_index * 4 + 3],
        )
    }

    fn is_rectangle(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> bool {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(mesh, blp_index);

        if p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00 {
            return false;
        }

        let n = (p10 - p00).cross(p01 - p00).normalize();

        if (p11 - p00).normalize().dot(n).abs() > 1e-5 {
            return false;
        }

        let p_center = (p00 + p01 + p10 + p11) * 0.25;
        let d2: [Float; 4] = [
            (p00 - p_center).length_squared(),
            (p01 - p_center).length_squared(),
            (p10 - p_center).length_squared(),
            (p11 - p_center).length_squared(),
        ];

        for i in 1..4 {
            if Float::abs(d2[i] - d2[0]) / d2[0] > 1e-4 {
                return false;
            }
        }

        true
    }

    fn intersect_blp(ray: &Ray, t_max: Float, p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> Option<BilinearIntersection> {
        let a = (p10 - p00).cross(p01 - p11).dot(ray.direction);
        let c = (p00 - ray.origin).cross(ray.direction.into()).dot(p01 - p00);
        let b = (p10 - ray.origin).cross(ray.direction.into()).dot(p11 - p10) - (a + c);

        let (u1, u2) = quadratic(a, b, c)?;

        let eps = gamma(10) * (
            ray.origin.abs().max_element() + ray.direction.abs().max_element()
            + p00.abs().max_element() + p10.abs().max_element()
            + p01.abs().max_element() + p11.abs().max_element()
        );

        let mut t = t_max;
        let mut u = 0.0;
        let mut v = 0.0;

        if (0.0..=1.0).contains(&u1) {
            let uo = lerp_float(p00, p10, u1);
            let ud = lerp_float(p01, p11, u1) - uo;

            let deltao = uo - ray.origin;
            let perp = ray.direction.cross(ud.into());
            let p2 = perp.length_squared();

            let v1 = Mat3::new(
                deltao.x, ray.direction.x, perp.x,
                deltao.y, ray.direction.y, perp.y,
                deltao.z, ray.direction.z, perp.z,
            ).determinant();

            let t1 = Mat3::new(
                deltao.x, ud.x, perp.x,
                deltao.y, ud.y, perp.y,
                deltao.z, ud.z, perp.z,
            ).determinant();

            if t1 > p2 * eps && 0.0 <= v1 && v1 <= p2 {
                u = u1;
                v = v1 / p2;
                t = t1 / p2;
            }
        }

        if (0.0..=1.0).contains(&u2) && u2 != u1 {
            let uo = lerp_float(p00, p10, u2);
            let ud = lerp_float(p01, p11, u2) - uo;

            let deltao = uo - ray.origin;
            let perp = ray.direction.cross(ud.into());
            let p2 = perp.length_squared();
            
            let v2 = Mat3::new(
                deltao.x, ray.direction.x, perp.x,
                deltao.y, ray.direction.y, perp.y,
                deltao.y, ray.direction.z, perp.z,
            ).determinant();

            let mut t2 = Mat3::new(
                deltao.x, ud.x, perp.x,
                deltao.y, ud.y, perp.y,
                deltao.z, ud.z, perp.z,
            ).determinant();

            t2 /= p2;

            if 0.0 <= v2 && v2 <= p2 && t > t2 && t2 > eps {
                t = t2;
                u = u2;
                v = v2 / p2;
            }
        }

        if t >= t_max {
            return None;
        }

        Some(BilinearIntersection {
            uv: Point2f::new(u, v),
            t,
        })
    }

    fn interaction_from_intersection(
        mesh: &Arc<BilinearPatchMesh>,
        blp_index: usize,
        uv: Point2f,
        time: Float,
        wo: Vec3f,
    ) -> SurfaceInteraction {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(mesh, blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(mesh, blp_index);

        let p = lerp_float(lerp_float(p00, p01, uv.y), lerp_float(p10, p11, uv.y), uv.x);
        let mut dpdu = Vec3f::from(lerp_float(p10, p11, uv.y) - lerp_float(p00, p01, uv.y));
        let mut dpdv = Vec3f::from(lerp_float(p01, p11, uv.x) - lerp_float(p00, p10, uv.x));

        let mut st = uv;
        let mut duds = 1.0;
        let mut dudt = 0.0;
        let mut dvds = 1.0;
        let mut dvdt = 0.0;

        if !mesh.uv.is_empty() {
            let uv00 = mesh.uv[v0];
            let uv10 = mesh.uv[v1];
            let uv01 = mesh.uv[v2];
            let uv11 = mesh.uv[v3];
            st = lerp_float(lerp_float(uv00, uv01, uv.y), lerp_float(uv10, uv11, uv.y), uv.x);

            let dstdu = lerp_float(uv10, uv11, uv.y) - lerp_float(uv00, uv01, uv.y);
            let dstdv = lerp_float(uv01, uv11, uv.x) - lerp_float(uv00, uv10, uv.x);

            duds = if dstdu.x.abs() < 1e-8 {
                0.0
            } else {
                1.0 / dstdu.x
            };
            dvds = if dstdv.x.abs() < 1e-8 {
                0.0
            } else {
                1.0 / dstdv.x
            };
            dudt = if dstdu.y.abs() < 1e-8 {
                0.0
            } else {
                1.0 / dstdu.y
            };
            dvdt = if dstdv.y.abs() < 1e-8 {
                0.0
            } else {
                1.0 / dstdv.y
            };

            let dpds = dpdu * duds + dpdv * dvds;
            let mut dpdt = dpdu * dudt + dpdv * dvdt;

            if dpds.cross(dpdt) != Vec3f::ZERO {
                if dpdu.cross(dpdv).dot(dpds.cross(dpdt)) < 0.0 {
                    dpdt = -dpdt;
                }

                debug_assert!(dpdu.cross(dpdv).normalize().dot(dpds.cross(dpdt).normalize()) > -1e3);
                dpdu = dpds;
                dpdv = dpdt;
            }
        }

        let d2pduu = Vec3f::ZERO;
        let d2pdvv = Vec3f::ZERO;
        let d2pduv = (p00 - p01) + (p11 - p10);

        let e1 = dpdu.dot(dpdu);
        let f1 = dpdu.dot(dpdv);
        let g1 = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdv).normalize();
        let e2 = n.dot(d2pduu);
        let f2 = n.dot(d2pduv);
        let g2 = n.dot(d2pdvv);

        let egf2 = Float::difference_of_products(e1, g1, f1, f1);
        let inv_egf2 = if egf2 != 0.0 {
            1.0 / egf2
        } else {
            0.0
        };

        let mut dndu = Normal3f::from((f1 * f2 - e2 * g1) * inv_egf2 * dpdu + (e2 * f1 - f2 * e1) * inv_egf2 * dpdv);
        let mut dndv = Normal3f::from((g2 * f1 - f2 * g1) * inv_egf2 * dpdu + (f2 * f1 - g2 * e1) * inv_egf2 * dpdv);

        let dnds = dndu * duds + dndv * dvds;
        let dndt = dndu * dudt + dndv * dvdt;

        dndu = dnds;
        dndv = dndt;

        let p_abs_sum = p00.abs() + p01.abs() + p10.abs() + p11.abs();
        let p_error = gamma(6) * p_abs_sum;
        
        let face_index = if mesh.face_indices.is_empty() {
            0
        } else {
            mesh.face_indices[blp_index]
        };

        let flip_normal = mesh.reverse_orientation ^ mesh.transform_swaps_handedness;
        let mut isect = SurfaceInteraction::new_with_face_index(
            Point3fi::from_errors(p, p_error),
            st,
            wo,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            flip_normal,
            face_index as i32,
        );

        if !mesh.n.is_empty() {
            let n00 = mesh.n[v0];
            let n10 = mesh.n[v1];
            let n01 = mesh.n[v2];
            let n11 = mesh.n[v3];
            let mut ns = lerp_float(lerp_float(n00, n01, uv.y), lerp_float(n10, n11, uv.y), uv.x);

            if ns.length_squared() > 0.0 {
                ns = ns.normalize();
                let mut dndu = lerp_float(n10, n11, uv.y) - lerp_float(n00, n01, uv.y);
                let mut dndv = lerp_float(n10, n11, uv.x) - lerp_float(n00, n01, uv.x);
                let dnds = dndu * duds + dndv * dvds;
                let dndt = dndu * dudt + dndv * dvdt;

                dndu = dnds;
                dndv = dndt;

                let r = Transform::from_rotation_delta(isect.interaction.n.into(), ns.into());
                isect.set_shading_geometry(ns, r.apply(dpdu), r.apply(dpdv), dndu, dndv, true);
            }
        }

        isect
    }
}

impl AbstractShape for BilinearPatch {
    fn bounds(&self) -> Bounds3f {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        Bounds3f::new(p00, p01).union(Bounds3f::new(p10, p11))
    }

    fn normal_bounds(&self) -> DirectionCone {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        if p00 == p10 || p10 == p11 || p11 == p01 || p01 == p00 {
            let dpdu = lerp_float(p10, p11, 0.5) - lerp_float(p00, p01, 0.5);
            let dpdv = lerp_float(p01, p11, 0.5) - lerp_float(p00, p10, 0.5);
            let n = dpdu.cross(dpdv).normalize();

            let n = if !self.mesh.n.is_empty() {
                let n00 = self.mesh.n[v0];
                let n10 = self.mesh.n[v0];
                let n01 = self.mesh.n[v0];
                let n11 = self.mesh.n[v0];
                let ns = (n00 + n10 + n01 + n11) * 0.25;
                n.facing(ns.into())
            } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
                -n
            } else {
                n
            };

            return DirectionCone::from_direction(n.into())
        }

        let n00 = (p10 - p00).cross(p01 - p00).normalize();
        let n00 = if !self.mesh.n.is_empty() {
            n00.facing(self.mesh.n[v0].into())
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            -n00
        } else {
            n00
        };

        let n10 = (p11 - p10).cross(p00 - p10).normalize();
        let n01 = (p00 - p01).cross(p11 - p01).normalize();
        let n11 = (p01 - p11).cross(p10 - p11).normalize();
        let (n10, n01, n11) = if !self.mesh.n.is_empty() {
            (
                n10.facing(self.mesh.n[v1].into()),
                n01.facing(self.mesh.n[v2].into()),
                n11.facing(self.mesh.n[v3].into()),
            )
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            (-n10, -n01, -n11)
        } else {
            (n10, n01, n11)
        };

        let n = (n00 + n10 + n01 + n11).normalize();
        let cos_theta = [n.dot(n00), n.dot(n10), n.dot(n01), n.dot(n11)].into_iter()
            .min_by(|a, b| a.partial_cmp(b).expect("unexpected NaN")).unwrap();
        DirectionCone::new(n.into(), Float::clamp(cos_theta, -1.0, 1.0))
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let intersection = BilinearPatch::intersect_blp(ray, t_max, p00, p10, p01, p11)?;

        let interaction = BilinearPatch::interaction_from_intersection(
            &self.mesh,
            self.blp_index,
            intersection.uv,
            ray.time,
            -ray.direction,
        );

        Some(ShapeIntersection {
            intr: interaction,
            t_hit: intersection.t,
        })
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let intersection = BilinearPatch::intersect_blp(ray, t_max, p00, p10, p01, p11);
        intersection.is_some()
    }

    fn area(&self) -> Float {
        self.area
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        // TODO: Handle image distributions for emission
        let (uv, pdf) = if BilinearPatch::is_rectangle(&self.mesh, self.blp_index) {
            (u, 1.0)
        } else {
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];

            let uv = sample_bilinear(u, &w);
            let pdf = bilinear_pdf(uv, &w);
            (uv, pdf)
        };

        let pu0 = lerp_float(p00, p10, uv.x);
        let pu1 = lerp_float(p10, p11, uv.y);
        let p = lerp_float(pu0, pu1, uv.x);
        let dpdu = pu1 - pu0;
        let dpdv = lerp_float(p01, p11, uv.x) - lerp_float(p00, p10, uv.x);

        if dpdu.length_squared() == 0.0 || dpdv.length_squared() == 0.0 {
            return None;
        }

        let mut st = uv;
        if !self.mesh.uv.is_empty() {
            let uv00 = self.mesh.uv[v0];
            let uv10 = self.mesh.uv[v1];
            let uv01 = self.mesh.uv[v2];
            let uv11 = self.mesh.uv[v3];

            st = lerp_float(lerp_float(uv00, uv01, uv.y), lerp_float(uv10, uv11, uv.y), uv.x);
        }

        let mut n = dpdu.cross(dpdv).normalize();

        if !self.mesh.n.is_empty() {
            let n00 = self.mesh.n[v0];
            let n10 = self.mesh.n[v1];
            let n01 = self.mesh.n[v2];
            let n11 = self.mesh.n[v3];
            let ns = lerp_float(lerp_float(n00, n01, uv.y), lerp_float(n10, n11, uv.y), uv.x);
            n = n.facing(ns.into());
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            n = -n;
        }

        let p_abs_sum = p00.abs() + p01.abs() + p10.abs() + p11.abs();
        let p_error = gamma(6) * p_abs_sum;

        Some(ShapeSample {
            intr: Interaction::new(
                Point3fi::from_errors(p, p_error),
                n.into(),
                st,
                Vec3f::ZERO,
                0.0,
            ),
            pdf: pdf / dpdu.cross(dpdv).length(),
        })
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        let mut uv = interaction.uv;
        if !self.mesh.uv.is_empty() {
            let uv00 = self.mesh.uv[v0];
            let uv10 = self.mesh.uv[v1];
            let uv01 = self.mesh.uv[v2];
            let uv11 = self.mesh.uv[v3];
            uv = invert_bilinear(uv, &[uv00, uv10, uv01, uv11]);
        }

        // TODO: Handle image distributions
        let pdf = if !BilinearPatch::is_rectangle(&self.mesh, self.blp_index) {
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];
            bilinear_pdf(uv, &w)
        } else {
            1.0
        };
        
        let pu0 = lerp_float(p00, p10, uv.y);
        let pu1 = lerp_float(p10, p11, uv.y);
        let dpdu = pu1 - pu0;
        let dpdv = lerp_float(p01, p11, uv.x) - lerp_float(p00, p10, uv.x);

        pdf / dpdu.cross(dpdv).length()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, mut u: Point2f) -> Option<ShapeSample> {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        let v00 = (p00 - ctx.p()).normalize();
        let v10 = (p10 - ctx.p()).normalize();
        let v01 = (p01 - ctx.p()).normalize();
        let v11 = (p11 - ctx.p()).normalize();

        // TODO: Handle image distributions
        if !Self::is_rectangle(&self.mesh, self.blp_index) || spherical_quad_area(v00, v10, v11, v01) <= Self::MIN_SPHERICAL_SAMPLE_AREA {
            let mut ss = self.sample(u).unwrap();
            ss.intr.time = ctx.time;
            let mut wi = ss.intr.position() - ctx.p();
            if wi.length_squared() == 0.0 {
                return None;
            }
            wi = wi.normalize();

            ss.pdf /= ss.intr.n.dot(-wi).abs() / (ctx.p() - ss.intr.position()).length_squared();
            if ss.pdf.is_infinite() {
                return None;
            }
            return Some(ss)
        }

        let mut pdf = 1.0;
        if ctx.ns != Normal3f::ZERO {
            let w = [
                Float::max(0.01, v00.dot(ctx.ns)),
                Float::max(0.01, v10.dot(ctx.ns)),
                Float::max(0.01, v01.dot(ctx.ns)),
                Float::max(0.01, v11.dot(ctx.ns)),
            ];

            u = sample_bilinear(u, &w);
            pdf = bilinear_pdf(u, &w);
        }

        let eu = p10 - p00;
        let ev = p01 - p00;
        let mut quad_pdf = 0.0;
        let p = sample_spherical_rectangle(ctx.p(), p00, eu.into(), ev.into(), u, Some(&mut quad_pdf));
        pdf *= quad_pdf;

        let uv = Point2f::new(
            (p - p00).dot(eu) / p10.distance_squared(p00),
            (p - p00).dot(ev) / p01.distance_squared(p00),
        );

        let mut n: Normal3f = eu.cross(ev).normalize().into();

        if !self.mesh.n.is_empty() {
            let n00 = self.mesh.n[v0];
            let n10 = self.mesh.n[v1];
            let n01 = self.mesh.n[v2];
            let n11 = self.mesh.n[v3];
            let ns = lerp_float(lerp_float(n00, n01, uv.y), lerp_float(n10, n11, uv.y), uv.x);
            n = n.facing(ns);
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            n = -n;
        }

        let mut st = uv;
        if !self.mesh.uv.is_empty() {
            let uv00 = self.mesh.uv[v0];
            let uv10 = self.mesh.uv[v1];
            let uv01 = self.mesh.uv[v2];
            let uv11 = self.mesh.uv[v3];
            st = lerp_float(lerp_float(uv00, uv01, uv.y), lerp_float(uv10, uv11, uv.y), uv.x);
        }

        Some(
            ShapeSample{
                intr: Interaction::new(
                    Point3fi::from(p),
                    n,
                    st,
                    Vec3f::ZERO,
                    ctx.time,
                ),
                pdf,
            }
        )
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);

        let ray = ctx.spawn_ray(wi);
        let Some(isect) = self.intersect(&ray, Float::INFINITY) else {
            return 0.0;
        };

        let v00 = (p00 - ctx.p()).normalize();
        let v10 = (p10 - ctx.p()).normalize();
        let v01 = (p01 - ctx.p()).normalize();
        let v11 = (p11 - ctx.p()).normalize();

        // TODO: Handle image distributions
        if !Self::is_rectangle(&self.mesh, self.blp_index) || spherical_quad_area(v00, v10, v11, v01) <= Self::MIN_SPHERICAL_SAMPLE_AREA {
            let pdf = self.pdf(&isect.intr.interaction) * (ctx.p().distance_squared(isect.intr.position())) 
                / isect.intr.interaction.n.dot(-wi).abs();
            if pdf.is_infinite() {
                0.0
            } else {
                pdf
            }
        } else {
            let pdf = 1.0 / spherical_quad_area(v00, v10, v11, v01);
            if ctx.ns != Normal3f::ZERO {
                let w = [
                    Float::max(0.01, v00.dot(ctx.ns)),
                    Float::max(0.01, v10.dot(ctx.ns)),
                    Float::max(0.01, v01.dot(ctx.ns)),
                    Float::max(0.01, v11.dot(ctx.ns)),
                ];
                let u = invert_spherical_rectangle_sample(ctx.p(), p00, (p10 - p00).into(), (p01 - p00).into(), isect.intr.position());
                bilinear_pdf(u, &w) * pdf
            } else {
                pdf
            }
        }
    }
}
