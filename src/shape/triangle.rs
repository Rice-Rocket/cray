use std::sync::Arc;

use crate::{bounds::Union, difference_of_products_float_vec, error, gamma, interaction::{Interaction, SurfaceInteraction}, math::numeric::DifferenceOfProducts, reader::{error::ParseResult, paramdict::ParameterDictionary, target::FileLoc}, sampling::{bilinear_pdf, invert_spherical_triangle_sample, sample_bilinear, sample_spherical_triangle, sample_uniform_triangle}, spherical_triangle_area, transform::Transform, Bounds3f, DirectionCone, Dot, Float, Normal3f, Point2f, Point3, Point3f, Point3fi, Ray, Vec2f, Vec3f};

use super::{mesh::TriangleMesh, AbstractShape, Shape, ShapeIntersection, ShapeSample, ShapeSampleContext};

#[derive(Debug, Clone)]
pub struct Triangle {
    mesh: Arc<TriangleMesh>,
    tri_index: i32,
}

impl Triangle {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 3e-4;
    const MAX_SPHERICAL_SAMPLE_AREA: Float = 6.22;

    pub fn create_mesh(
        render_from_object: &Transform,
        reverse_orientation: bool,
        parameters: &mut ParameterDictionary,
        loc: &FileLoc,
    ) -> ParseResult<TriangleMesh> {
        let mut vi = parameters.get_int_array("indices")?;
        let p = parameters.get_point3f_array("P")?;
        let uvs = parameters.get_point2f_array("uv")?;

        if vi.is_empty() {
            if p.len() == 3 {
                vi = vec![0, 1, 2];
            } else {
                error!(loc, MissingParameter, "vertex indices 'indices' not provided with trianglemesh shape");
            }
        } else if vi.len() % 3 != 0 {
            error!(loc, InvalidValueCount, "number of vertex indices 'indices' not multiple of 3 as expected");
            // TODO: Could just pop excess and warn
        }

        if p.is_empty() {
            error!(loc, MissingParameter, "vertex positions 'P' not provided with trianglemesh shape");
        }

        if !uvs.is_empty() && uvs.len() != p.len() {
            error!(loc, InvalidValueCount, "number of vertex positions 'P' and vertex UVs 'uv' do not match");
            // TODO: Could just discard uvs instead of panicing + warn
        }

        let s = parameters.get_vector3f_array("S")?;
        if !s.is_empty() && s.len() != p.len() {
            error!(loc, InvalidValueCount, "number of vertex positions 'P' and vertex tangents 'S' do not match");
            // TODO: Could just discard instead of panicing + warn
        }

        let n = parameters.get_normal3f_array("N")?;
        if !n.is_empty() && n.len() != p.len() {
            error!(loc, InvalidValueCount, "number of vertex positions 'P' and vertex normals 'N' do not match");
            // TODO: Could just discard instead of panicing + warn
        }

        for v in vi.iter() {
            if *v as usize >= p.len() {
                error!(loc, InvalidValue, "vertex indices {} out of bounds 'P' array length {}", v, p.len());
            }
        }

        let face_indices = parameters.get_int_array("faceIndices")?;
        if !face_indices.is_empty() && face_indices.len() != vi.len() / 3 {
            error!(loc, InvalidValueCount, "number of face indices 'faceIndices' and vertex indices 'indices' do not match");
            // TODO: Could just discard instead of panicing + warn
        }

        Ok(TriangleMesh::new(
            render_from_object,
            reverse_orientation,
            vi.into_iter().map(|i| i as usize).collect(),
            p,
            s,
            n,
            uvs,
            face_indices.into_iter().map(|i| i as usize).collect(),
        ))
    }
    
    pub fn create_triangles(mesh: Arc<TriangleMesh>) -> Vec<Arc<Shape>> {
        let mut tris = Vec::with_capacity(mesh.n_triangles);
        for i in 0..mesh.n_triangles {
            tris.push(Arc::new(Shape::Triangle(Box::new(Triangle::new(
                mesh.clone(),
                i as i32,
            )))));
        }
        tris
    }

    pub fn new(mesh: Arc<TriangleMesh>, tri_index: i32) -> Triangle {
        Triangle { mesh, tri_index }
    }

    pub fn get_mesh(&self) -> &Arc<TriangleMesh> {
        &self.mesh
    }

    fn get_points(&self) -> (Point3f, Point3f, Point3f) {
        let v = self.get_vertex_indices();
        let p0 = self.mesh.p[v[0]];
        let p1 = self.mesh.p[v[1]];
        let p2 = self.mesh.p[v[2]];
        (p0, p1, p2)
    }

    fn get_vertex_indices(&self) -> &[usize] {
        &self.mesh.vertex_indices
            [(3 * self.tri_index as usize)..((3 * self.tri_index as usize) + 3)]
    }

    pub fn solid_angle(&self, p: Point3f) -> Float {
        let (p0, p1, p2) = self.get_points();
        spherical_triangle_area(
            (p0 - p).normalize().into(),
            (p1 - p).normalize().into(),
            (p2 - p).normalize().into(),
        )
    }

    pub fn intersect_triangle(
        ray: &Ray,
        t_max: Float,
        p0: Point3f,
        p1: Point3f,
        p2: Point3f,
    ) -> Option<TriangleIntersection> {
        if (p2 - p0).cross(p1 - p0).length_squared() == 0.0 {
            return None;
        }

        let mut p0t = p0 - ray.origin;
        let mut p1t = p1 - ray.origin;
        let mut p2t = p2 - ray.origin;

        let kz = ray.direction.abs().max_element_index();
        let mut kx = kz + 1;
        if kx == 3 {
            kx = 0;
        }

        let mut ky = kx + 1;
        if ky == 3 {
            ky = 0;
        }

        let d = ray.direction.permute((kx, ky, kz));
        p0t = p0t.permute((kx, ky, kz));
        p1t = p1t.permute((kx, ky, kz));
        p2t = p2t.permute((kx, ky, kz));

        let sx = -d.x / d.z;
        let sy = -d.y / d.z;
        let sz = 1.0 / d.z;
        p0t.x += sx * p0t.z;
        p0t.y += sy * p0t.z;
        p1t.x += sx * p1t.z;
        p1t.y += sy * p1t.z;
        p2t.x += sx * p2t.z;
        p2t.y += sy * p2t.z;

        let mut e0 = Float::difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
        let mut e1 = Float::difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
        let mut e2 = Float::difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

        #[cfg(not(feature = "use_f64"))]
        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            let p2txp1ty = p2t.x as f64 * p1t.y as f64;
            let p2typ1tx = p2t.y as f64 * p1t.x as f64;
            e0 = (p2typ1tx - p2txp1ty) as f32;
            let p0txp2ty = p0t.x as f64 * p2t.y as f64;
            let p0typ2tx = p0t.y as f64 * p2t.x as f64;
            e1 = (p0typ2tx - p0txp2ty) as f32;
            let p1txp0ty = p1t.x as f64 * p0t.y as f64;
            let p1typ0tx = p1t.y as f64 * p0t.x as f64;
            e2 = (p1typ0tx - p1txp0ty) as f32;
        }

        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return None;
        }
        let det = e0 + e1 + e2;
        if det == 0.0 {
            return None;
        }

        p0t.z *= sz;
        p1t.z *= sz;
        p2t.z *= sz;
        let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if det < 0.0 && (t_scaled >= 0.0 || t_scaled < t_max * det)
        || det > 0.0 && (t_scaled <= 0.0 || t_scaled > t_max * det) {
            return None;
        }

        let inv_det = 1.0 / det;
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;
        let t = t_scaled * inv_det;
        debug_assert!(!t.is_nan());

        let max_zt = Vec3f::new(p0t.z, p1t.z, p2t.z).abs().max_element();
        let delta_z = gamma(3) * max_zt;

        let max_xt = Vec3f::new(p0t.x, p1t.x, p2t.x).abs().max_element();
        let max_yt = Vec3f::new(p0t.y, p1t.y, p2t.y).abs().max_element();

        let delta_x = gamma(5) * (max_xt + max_zt);
        let delta_y = gamma(5) * (max_yt + max_zt);

        let delta_e = 2.0 * (gamma(2) * max_xt * max_yt + delta_y * max_xt + delta_x * max_yt);

        let max_e = Vec3f::new(e0, e1, e2).abs().max_element();
        let delta_t = 3.0
            * (gamma(3) * max_e * max_zt + delta_e * max_zt + delta_z * max_e)
            * Float::abs(inv_det);
        if t <= delta_t {
            return None;
        }

        Some(TriangleIntersection { b0, b1, b2, t })
    }

    fn interaction_from_intersection(
        &self,
        ti: TriangleIntersection,
        time: Float,
        wo: Vec3f,
    ) -> SurfaceInteraction {
        let (p0, p1, p2) = self.get_points();
        let v = self.get_vertex_indices();

        // Compute triangle partial derivatives
        // Computes deltas and matrix determinant for triangle partial derivatives
        // Get triangle texture coordinates in uv array
        let uv = if self.mesh.uv.is_empty() {
            [Point2f::ZERO, Point2f::new(1.0, 0.0), Point2f::ONE]
        } else {
            [self.mesh.uv[v[0]], self.mesh.uv[v[1]], self.mesh.uv[v[2]]]
        };
        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = Float::difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

        let degenerate_uv = determinant.abs() < 1e-9;
        let (dpdu, dpdv) = if !degenerate_uv {
            let inv_det = 1.0 / determinant;
            let dpdu = difference_of_products_float_vec(duv12[1], dp02.into(), duv02[1], dp12.into()) * inv_det;
            let dpdv = difference_of_products_float_vec(duv02[0], dp12.into(), duv12[0], dp02.into()) * inv_det;
            (dpdu, dpdv)
        } else {
            (Vec3f::ZERO, Vec3f::ZERO)
        };

        let (dpdu, dpdv) = if degenerate_uv || dpdu.cross(dpdv).length_squared() == 0.0 {
            let mut ng = (p2 - p0).cross(p1 - p0);
            if ng.length_squared() == 0.0 {
                let v1 = p2 - p0;
                let v2 = p1 - p0;
                #[cfg(not(feature = "use_f64"))]
                {
                    ng = Point3::<f64>::from(v1).cross(Point3::<f64>::from(v2)).into();
                }
                #[cfg(feature = "use_f64")]
                {
                    ng = v1.cross(v2);
                }
                debug_assert_ne!(ng.length_squared(), 0.0);
            }
            ng.normalize().coordinate_system()
        } else {
            (dpdu, dpdv)
        };

        let p_hit: Point3f = ti.b0 * p0 + ti.b1 * Vec3f::from(p1) + ti.b2 * Vec3f::from(p2);
        let uv_hit: Point2f = ti.b0 * uv[0] + ti.b1 * Vec2f::from(uv[1]) + ti.b2 * Vec2f::from(uv[2]);

        let flip_normal = self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness;

        let p_abs_sum = (ti.b0 * p0).abs()
            + Vec3f::from((ti.b1 * p1).abs())
            + Vec3f::from((ti.b2 * p2).abs());
        let p_error = gamma(7) * Vec3f::from(p_abs_sum);

        let mut isect = SurfaceInteraction::new(
            Point3fi::from_errors(p_hit, p_error.into()),
            uv_hit,
            wo,
            dpdu,
            dpdv,
            Normal3f::ZERO,
            Normal3f::ZERO,
            time,
            flip_normal,
        );

        isect.face_index = if self.mesh.face_indices.is_empty() {
            0
        } else {
            self.mesh.face_indices[self.tri_index as usize] as i32
        };

        isect.shading.n = dp02.cross(dp12).normalize().into();
        isect.interaction.n = isect.shading.n;
        if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            isect.shading.n = -isect.shading.n;
            isect.interaction.n = -isect.interaction.n;
        }

        if !self.mesh.n.is_empty() || !self.mesh.s.is_empty() {
            let ns = if self.mesh.n.is_empty() {
                isect.interaction.n
            } else {
                let n = ti.b0 * self.mesh.n[v[0]]
                    + ti.b1 * self.mesh.n[v[1]]
                    + ti.b2 * self.mesh.n[v[2]];
                if n.length_squared() > 0.0 {
                    n.normalize()
                } else {
                    isect.interaction.n
                }
            };

            let ss = if self.mesh.s.is_empty() {
                isect.dpdu
            } else {
                let s = ti.b0 * self.mesh.s[v[0]]
                    + ti.b1 * self.mesh.s[v[1]]
                    + ti.b2 * self.mesh.s[v[2]];
                if s.length_squared() == 0.0 {
                    isect.dpdu
                } else {
                    s
                }
            };

            let ts = Vec3f::from(ns).cross(ss);
            let (ss, ts) = if ts.length_squared() > 0.0 {
                (ts.cross(ns.into()), ts)
            } else {
                Vec3f::from(ns).coordinate_system()
            };

            let (dndu, dndv) = if self.mesh.n.is_empty() {
                (Normal3f::ZERO, Normal3f::ZERO)
            } else {
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];

                let determinant = Float::difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);
                let degenerate_uv = determinant.abs() < 1e-9;
                if degenerate_uv {
                    let dn = Vec3f::from(self.mesh.n[v[2]] - self.mesh.n[v[0]])
                        .cross(Vec3f::from(self.mesh.n[v[1]] - self.mesh.n[v[0]]));
                    if dn.length_squared() == 0.0 {
                        (Normal3f::ZERO, Normal3f::ZERO)
                    } else {
                        let (dnu, dnv) = dn.coordinate_system();
                        (dnu.into(), dnv.into())
                    }
                } else {
                    let inv_det = 1.0 / determinant;
                    let dn1 = self.mesh.n[v[0]] - self.mesh.n[v[2]];
                    let dn2 = self.mesh.n[v[1]] - self.mesh.n[v[2]];
                    (
                        (difference_of_products_float_vec(
                            duv12[1],
                            dn1.into(),
                            duv02[1],
                            dn2.into(),
                        ) * inv_det)
                            .into(),
                        (difference_of_products_float_vec(
                            duv02[0],
                            dn2.into(),
                            duv12[0],
                            dn1.into(),
                        ) * inv_det)
                            .into(),
                    )
                }
            };

            isect.set_shading_geometry(ns, ss, ts, dndu, dndv, true)
        }

        isect
    }
}

impl AbstractShape for Triangle {
    fn bounds(&self) -> Bounds3f {
        let (p0, p1, p2) = self.get_points();
        Bounds3f::new(p0, p1).union(p2)
    }

    fn normal_bounds(&self) -> DirectionCone {
        let v = self.get_vertex_indices();
        let (p0, p1, p2) = self.get_points();
        let n = (p1 - p0).cross(p2 - p0).normalize();
        let n = if !self.mesh.n.is_empty() {
            let ns = self.mesh.n[v[0]] + self.mesh.n[v[1]] + self.mesh.n[v[2]];
            n.facing(ns.into())
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            -n
        } else {
            n
        };
        DirectionCone::from_direction(n.into())
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (p0, p1, p2) = self.get_points();
        let tri_isect = Triangle::intersect_triangle(ray, t_max, p0, p1, p2)?;
        let t_hit = tri_isect.t;
        let intr = self.interaction_from_intersection(tri_isect, ray.time, -ray.direction);
        Some(ShapeIntersection { intr, t_hit })
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        let (p0, p1, p2) = self.get_points();
        let tri_isect = Triangle::intersect_triangle(ray, t_max, p0, p1, p2);
        tri_isect.is_some()
    }

    fn area(&self) -> Float {
        let (p0, p1, p2) = self.get_points();
        0.5 * (p1 - p0).cross(p2 - p0).length()
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        let (p0, p1, p2) = self.get_points();
        let v = self.get_vertex_indices();

        let (b0, b1, b2) = sample_uniform_triangle(u);
        let p = b0 * p0 + (b1 * p1) + (b2 * p2);

        let n: Normal3f = (p1 - p0).cross(p2 - p0).normalize().into();
        let n = if self.mesh.n.is_empty() {
            -n
        } else {
            let ns: Normal3f =
                b0 * self.mesh.n[v[0]] + b1 * self.mesh.n[v[1]] + b2 * self.mesh.n[v[2]];
            n.facing(ns)
        };

        let (uv0, uv1, uv2) = if self.mesh.uv.is_empty() {
            (Point2f::ZERO, Point2f::new(1.0, 0.0), Point2f::new(0.0, 1.0))
        } else {
            (self.mesh.uv[v[0]], self.mesh.uv[v[1]], self.mesh.uv[v[2]])
        };

        let uv_sample: Point2f = b0 * uv0 + Vec2f::from(b1 * uv1) + Vec2f::from(b2 * uv2);

        let p_abs_sum = (b0 * p0).abs() + (b1 * p1).abs() + (b2 * p2).abs();
        let p_error = gamma(6) * p_abs_sum;

        Some(ShapeSample {
            intr: Interaction::new(
                Point3fi::from_errors(p, p_error),
                n,
                uv_sample,
                Default::default(),
                Default::default(),
            ),
            pdf: 1.0 / self.area(),
        })
    }

    fn pdf(&self, _interaction: &Interaction) -> Float {
        1.0 / self.area()
    }

    fn sample_with_context(
        &self,
        ctx: &ShapeSampleContext,
        u: Point2f,
    ) -> Option<ShapeSample> {
        let solid_angle = self.solid_angle(ctx.p());
        if !(Self::MIN_SPHERICAL_SAMPLE_AREA..=Self::MAX_SPHERICAL_SAMPLE_AREA).contains(&solid_angle)
        {
            let ss = self.sample(u);
            debug_assert!(ss.is_some());
            let mut ss = ss.expect("Expected sample to succeed");
            ss.intr.time = ctx.time;
            let wi = ss.intr.position() - ctx.p();
            if wi.length_squared() == 0.0 {
                return None;
            }
            let wi = wi.normalize();

            ss.pdf /= ss.intr.n.dot(-wi).abs() / ctx.p().distance_squared(ss.intr.position());
            if ss.pdf.is_infinite() {
                return None;
            }
            return Some(ss);
        }

        let (p0, p1, p2) = self.get_points();
        let v = self.get_vertex_indices();

        let pdf = if ctx.ns != Normal3f::ZERO {
            let rp = ctx.p();
            let wi = [
                (p0 - rp).normalize(),
                (p1 - rp).normalize(),
                (p2 - rp).normalize(),
            ];
            let w = [
                Float::max(0.01, ctx.ns.dot(wi[1]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[1]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[0]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[2]).abs()),
            ];
            let u = sample_bilinear(u, &w);
            debug_assert!(u[0] >= 0.0 && u[0] <= 1.0 && u[1] >= 0.0 && u[1] <= 1.0);
            bilinear_pdf(u, &w)
        } else {
            1.0
        };

        let (b, tri_pdf) = sample_spherical_triangle(&[p0, p1, p2], ctx.p(), u);
        if tri_pdf == 0.0 {
            return None;
        }
        let pdf = pdf * tri_pdf;

        let p_abs_sum = (b[0] * p0).abs() + (b[1] * p1).abs() + ((1.0 - b[0] - b[1]) * p2).abs();
        let p_error = gamma(6) * p_abs_sum;

        let p = b[0] * p0 + (b[1] * p1) + (b[2] * p2);
        let n: Normal3f = (p1 - p0).cross(p2 - p0).normalize().into();
        let n = if !self.mesh.n.is_empty() {
            let ns = b[0] * self.mesh.n[v[0]] + b[1] * self.mesh.n[v[1]] + b[2] * self.mesh.n[v[2]];
            n.facing(ns)
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            -n
        } else {
            n
        };

        let uv = if self.mesh.uv.is_empty() {
            [Point2f::ZERO, Point2f::new(1.0, 0.0), Point2f::ONE]
        } else {
            [self.mesh.uv[v[0]], self.mesh.uv[v[1]], self.mesh.uv[v[2]]]
        };

        let uv_sample: Point2f = b[0] * uv[0] + Vec2f::from(b[1] * uv[1]) + Vec2f::from(b[2] * uv[2]);

        Some(ShapeSample {
            intr: Interaction::new(
                Point3fi::from_errors(p, p_error),
                n,
                uv_sample,
                Vec3f::default(),
                ctx.time,
            ),
            pdf,
        })
    }

    fn pdf_with_context(
        &self,
        ctx: &ShapeSampleContext,
        wi: Vec3f,
    ) -> Float {
        let solid_angle = self.solid_angle(ctx.p());
        if !(Self::MIN_SPHERICAL_SAMPLE_AREA..=Self::MAX_SPHERICAL_SAMPLE_AREA).contains(&solid_angle)
        {
            let ray = ctx.spawn_ray(wi);
            let Some(isect) = self.intersect(&ray, Float::INFINITY) else { return 0.0 };

            let pdf = (1.0 / self.area())
                / (isect.intr.interaction.n.dot(-wi).abs()
                    / ctx.p().distance_squared(isect.intr.position()));
            if pdf.is_infinite() {
                return 0.0;
            }
            return pdf;
        }
        let mut pdf = 1.0 / solid_angle;
        if ctx.ns != Normal3f::ZERO {
            let (p0, p1, p2) = self.get_points();
            let u = invert_spherical_triangle_sample(&[p0, p1, p2], ctx.p(), wi);
            let rp = ctx.p();
            let wi = [
                (p0 - rp).normalize(),
                (p1 - rp).normalize(),
                (p2 - rp).normalize(),
            ];
            let w = [
                Float::max(0.01, ctx.ns.dot(wi[1]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[1]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[0]).abs()),
                Float::max(0.01, ctx.ns.dot(wi[2]).abs()),
            ];
            pdf *= bilinear_pdf(u, &w);
        }

        pdf
    }
}

#[derive(Debug, Clone)]
pub struct TriangleIntersection {
    b0: Float,
    b1: Float,
    b2: Float,
    t: Float,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use float_cmp::assert_approx_eq;

    use crate::{sampler::{AbstractSampler, IndependentSampler, Sampler}, shape::{mesh::TriangleMesh, AbstractShape, Shape, ShapeSampleContext}, transform::Transform, Float, Mat4, Normal3f, Point3f, Point3fi, Ray, Vec3f};

    use super::Triangle;

    #[test]
    fn test_triangle_intersect() {
        let vertices = vec![
            Point3f::new(1.0, 2.0, 1.0),
            Point3f::new(-1.0, 2.0, 1.0),
            Point3f::new(-1.0, 2.0, -1.0),
            Point3f::new(1.0, 2.0, -1.0),
        ];

        let indices = vec![0, 1, 2, 2, 3, 0];

        let mesh = Arc::new(TriangleMesh::new(
            &Transform::new_with_inverse(Mat4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, -1.0,
                0.0, 0.0, 1.0, -6.8,
                0.0, 0.0, 0.0, 1.0,
            )),
            false,
            indices,
            vertices,
            vec![],
            vec![],
            vec![],
            vec![],
        ));

        let tris = Triangle::create_triangles(mesh);

        let ray = Ray::new_with_time(Point3f::ZERO, Vec3f::new(-0.03684295, 0.15507203, -0.9872159), 0.5854021);
        if let Shape::Triangle(tri) = tris[0].as_ref() {
            let (p0, p1, p2) = tri.get_points();
            let isect = Triangle::intersect_triangle(&ray, Float::INFINITY, p0, p1, p2);

            assert!(isect.is_some());
            let isect = isect.unwrap();

            assert_approx_eq!(Float, isect.t, 6.448617);
            assert_approx_eq!(Float, isect.b0, 0.38120696);
            assert_approx_eq!(Float, isect.b1, 0.3357048);
            assert_approx_eq!(Float, isect.b2, 0.28308824);

            let isect = Triangle::intersect(tri, &ray, Float::INFINITY);

            assert!(isect.is_some());
            let isect = isect.unwrap();

            assert_approx_eq!(Float, isect.t_hit, 6.448617);
        }

        let p0 = Point3f::new(1.0, 1.0, -5.8);
        let p1 = Point3f::new(-1.0, 1.0, -5.8);
        let p2 = Point3f::new(-1.0, 1.0, -7.8);
        let isect = Triangle::intersect_triangle(&ray, Float::INFINITY, p0, p1, p2);

        assert!(isect.is_some());
        let isect = isect.unwrap();

        assert_approx_eq!(Float, isect.t, 6.448617);
        assert_approx_eq!(Float, isect.b0, 0.38120696);
        assert_approx_eq!(Float, isect.b1, 0.3357048);
        assert_approx_eq!(Float, isect.b2, 0.28308824);
    }
}
