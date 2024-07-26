use crate::{gamma, interaction::{Interaction, SurfaceInteraction}, numeric::DifferenceOfProducts as _, reader::{paramdict::ParameterDictionary, target::FileLoc}, safe, sampling::sample_uniform_sphere, spherical_direction, sqr, to_radians, transform::Transform, Bounds3f, DirectionCone, Float, Frame, Interval, Normal3f, NumericFloat, Point2f, Point3f, Point3fi, Ray, Vec3f, Vec3fi, PI, TAU};

use super::{QuadricIntersection, ShapeIntersection, AbstractShape, ShapeSample, ShapeSampleContext};

#[derive(Debug, Clone)]
pub struct Sphere {
    radius: Float,
    z_min: Float,
    z_max: Float,
    theta_z_min: Float,
    theta_z_max: Float,
    phi_max: Float,
    render_from_object: Transform,
    object_from_render: Transform,
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

impl Sphere {
    pub fn create(
        render_from_object: Transform,
        object_from_render: Transform,
        reverse_orientation: bool,
        parameters: &mut ParameterDictionary,
        _loc: &FileLoc,
    ) -> Sphere {
        let radius = parameters.get_one_float("radius", 1.0);
        let z_min = parameters.get_one_float("zmin", -radius);
        let z_max = parameters.get_one_float("zmax", radius);
        let phi_max = parameters.get_one_float("phimax", 360.0);

        Sphere::new(
            render_from_object,
            object_from_render,
            reverse_orientation,
            radius,
            z_min,
            z_max,
            phi_max,
        )
    }

    pub fn new(
        render_from_object: Transform,
        object_from_render: Transform,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Sphere {
        Sphere {
            radius,
            z_min: Float::clamp(Float::min(z_min, z_max), -radius, radius),
            z_max: Float::clamp(Float::max(z_min, z_max), -radius, radius),
            theta_z_min: Float::acos(Float::clamp(Float::min(z_min, z_max) / radius, -1.0, 1.0)),
            theta_z_max: Float::acos(Float::clamp(Float::max(z_min, z_max) / radius, -1.0, 1.0)),
            phi_max: to_radians(Float::clamp(phi_max, 0.0, 360.0)),
            render_from_object,
            object_from_render,
            reverse_orientation,
            transform_swaps_handedness: render_from_object.swaps_handedness(),
        }
    }
}

impl Sphere {
    pub fn basic_intersect(&self, ray: Ray, t_max: Float) -> Option<QuadricIntersection> {
        let oi = self.object_from_render * Point3fi::from(ray.origin);
        let di = self.object_from_render * Vec3fi::from(ray.direction);

        let a = di.x.sqr() + di.y.sqr() + di.z.sqr();
        let b = 2.0 * (di.x * oi.x + di.y * oi.y + di.z * oi.z);
        let c = oi.x.sqr() + oi.y.sqr() + oi.z.sqr() - Interval::from(self.radius).sqr();

        let v = Vec3fi::from(oi - b / (2.0 * a) * di);
        let length = v.length();
        
        let discrim = 4.0 * a * (Interval::from(self.radius) + length) * (Interval::from(self.radius) - length);
        if discrim.lower_bound() < 0.0 {
            return None;
        }

        let root_discrim = discrim.nsqrt();
        let q = if Float::from(b) < 0.0 {
            -0.5 * (b - root_discrim)
        } else {
            -0.5 * (b + root_discrim)
        };

        let t0 = q / a;
        let t1 = c / q;

        let (t0, t1) = if t0.lower_bound() > t1.lower_bound() {
            (t1, t0)
        } else {
            (t0, t1)
        };

        if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
            return None;
        }

        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_shape_hit.upper_bound() > t_max {
                return None;
            }
        }

        let mut p_hit: Point3f = Point3f::from(oi) + Float::from(t_shape_hit) * Point3f::from(Vec3f::from(di));
        p_hit = p_hit * (self.radius / p_hit.distance(Point3f::ZERO));

        if p_hit.x == 0.0 && p_hit.y == 0.0 {
            p_hit.x = 1e-5 * self.radius;
        }

        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += TAU;
        }

        if (self.z_min > -self.radius && p_hit.z < self.z_min)
        || (self.z_max < self.radius && p_hit.z > self.z_max)
        || phi > self.phi_max {
            if t_shape_hit == t1 {
                return None;
            }

            if t1.upper_bound() > t_max {
                return None;
            }

            t_shape_hit = t1;

            p_hit = Point3f::from(oi) + Float::from(t_shape_hit) * Point3f::from(Vec3f::from(di));
            p_hit = p_hit * (self.radius / p_hit.distance(Point3f::ZERO));

            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }

            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += TAU;
            }

            if (self.z_min > -self.radius && p_hit.z < self.z_min)
            || (self.z_max < self.radius && p_hit.z > self.z_max)
            || phi > self.phi_max {
                return None;
            }
        }

        Some(QuadricIntersection {
            t_hit: Float::from(t_shape_hit),
            p_obj: p_hit,
            phi,
        })
    }

    pub fn interaction_from_intersection(
        &self,
        isect: &QuadricIntersection,
        wo: Vec3f,
        time: Float,
    ) -> SurfaceInteraction {
        let p_hit = isect.p_obj;
        let phi = isect.phi;

        let u = phi / self.phi_max;
        let cos_theta = p_hit.z / self.radius;
        let theta = safe::acos(cos_theta);
        let v = (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min);

        let z_radius = Float::sqrt(sqr(p_hit.x) + sqr(p_hit.y));
        let cos_phi = p_hit.x / z_radius;
        let sin_phi = p_hit.y / z_radius;
        let dpdu = Vec3f::new(-self.phi_max * p_hit.y, self.phi_max * p_hit.x, 0.0);
        let sin_theta = safe::sqrt(1.0 - sqr(cos_theta));
        let dpdv = Vec3f::new(p_hit.z * cos_phi, p_hit.z * sin_phi, -self.radius * sin_theta)
            * (self.theta_z_max - self.theta_z_min);

        let d2pduu = Vec3f::new(p_hit.x, p_hit.y, 0.0) * -self.phi_max * self.phi_max;
        let d2pduv = Vec3f::new(-sin_phi, cos_phi, 0.0) * (self.theta_z_max - self.theta_z_min) * p_hit.z * self.phi_max;
        let d2pdvv = Vec3f::from(p_hit) * -sqr(self.theta_z_max - self.theta_z_min);
        let e1 = dpdu.dot(dpdu);
        let f1 = dpdu.dot(dpdv);
        let g1 = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdv).normalize();
        let e = n.dot(d2pduu);
        let f = n.dot(d2pduv);
        let g = n.dot(d2pdvv);

        let egf2 = Float::difference_of_products(e1, g1, f1, f1);
        let inv_egf2 = if egf2 == 0.0 { 0.0 } else { 1.0 / egf2 };
        let dndu = Normal3f::from((f * f1 - e * g1) * inv_egf2 * dpdu + (e * f1 - f * e1) * inv_egf2 * dpdv);
        let dndv = Normal3f::from((g * f1 - f * g1) * inv_egf2 * dpdu + (f * f1 - g * e1) * inv_egf2 * dpdv);

        let p_error = gamma(5) * Vec3f::from(p_hit).abs();

        let flip_normal = self.reverse_orientation ^ self.transform_swaps_handedness;
        let wo_object = self.object_from_render * wo;

        let si = SurfaceInteraction::new(
            Point3fi::from_errors(p_hit, p_error.into()),
            Point2f::new(u, v),
            wo_object,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            flip_normal,
        );

        self.render_from_object * si
    }
}

impl AbstractShape for Sphere {
    fn bounds(&self) -> Bounds3f {
        self.render_from_object * Bounds3f::new(
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        )
    }

    fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::entire_sphere()
    }

    fn intersect(&self, ray: Ray, t_max: Float) -> Option<ShapeIntersection> {
        let isect = self.basic_intersect(ray, t_max)?;
        let intr = self.interaction_from_intersection(&isect, -ray.direction, ray.time);
        Some(ShapeIntersection {
            intr,
            t_hit: isect.t_hit,
        })
    }

    fn intersect_predicate(&self, ray: Ray, t_max: Float) -> bool {
        self.basic_intersect(ray, t_max).is_some()
    }

    fn area(&self) -> Float {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        let mut p_obj = Point3f::ZERO + self.radius * sample_uniform_sphere(u);
        p_obj *= self.radius / p_obj.distance(Point3f::ZERO);
        let p_obj_err = gamma(5) * Vec3f::from(p_obj).abs();

        let n_obj = Normal3f::from(p_obj);
        let n = (self.render_from_object * n_obj).normalize() * if self.reverse_orientation { -1.0 } else { 1.0 };

        let theta = safe::acos(p_obj.z / self.radius);
        let mut phi = Float::atan2(p_obj.y, p_obj.x);
        if phi < 0.0 { phi += TAU };
        let uv = Point2f::new(phi / self.phi_max, (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min));

        let pi = self.render_from_object * Point3fi::from_errors(p_obj, p_obj_err.into());

        Some(ShapeSample {
            intr: Interaction::new(pi, n, uv, Vec3f::default(), 0.0),
            pdf: 1.0 / self.area(),
        })
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        1.0 / self.area()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        let p_center = self.render_from_object * Point3f::ZERO;
        let p_origin = ctx.offset_ray_origin_pt(p_center);

        if p_origin.distance_squared(p_center) <= sqr(self.radius) {
            let mut ss = self.sample(u).expect("sphere sample() failed");
            ss.intr.time = ctx.time;
            let wi = ss.intr.position() - ctx.p();
            if wi.length_squared() == 0.0 { return None };
            let wi = wi.normalize();

            ss.pdf /= ss.intr.n.dot(-Normal3f::from(wi)).abs() / ctx.p().distance_squared(ss.intr.position());
            if ss.pdf.is_infinite() { return None };

            return Some(ss);
        }

        let sin_theta_max = self.radius / ctx.p().distance(p_center);
        let sin2_theta_max = sqr(sin_theta_max);
        let cos_theta_max = safe::sqrt(1.0 - sin2_theta_max);
        let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;

        let mut cos_theta = (cos_theta_max - 1.0) * u.x + 1.0;
        let mut sin2_theta = 1.0 - sqr(cos_theta);
        if sin2_theta_max < 0.00068523 /* sin^2(1.5 deg) */ {
            sin2_theta = sin2_theta_max * u.x;
            cos_theta = Float::sqrt(1.0 - sin2_theta);
            one_minus_cos_theta_max = sin2_theta_max / 2.0;
        }

        let cos_alpha = sin2_theta / sin_theta_max + cos_theta * safe::sqrt(1.0 - sin2_theta / sqr(sin_theta_max));
        let sin_alpha = safe::sqrt(1.0 - sqr(cos_alpha));

        let phi = u.y * TAU;
        let w = spherical_direction(sin_alpha, cos_alpha, phi);
        let sampling_frame = Frame::from_z((p_center - ctx.p()).normalize().into());
        let n = Normal3f::from(sampling_frame.from_local(-w)) * if self.reverse_orientation { -1.0 } else { 1.0 };
        let p = p_center + Vec3f::from(n) * self.radius;

        let p_err = gamma(5) * Vec3f::from(p).abs();

        let p_obj = self.object_from_render * p;
        let theta = safe::acos(p_obj.z / self.radius);
        let mut sphere_phi = Float::atan2(p_obj.y, p_obj.x);
        if sphere_phi < 0.0 { sphere_phi += TAU };

        let uv = Point2f::new(sphere_phi / self.phi_max, (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min));

        Some(ShapeSample {
            intr: Interaction {
                pi: Point3fi::from_errors(p, p_err.into()),
                time: ctx.time,
                wo: Vec3f::default(),
                n,
                uv,
            },
            pdf: 1.0 / (TAU * one_minus_cos_theta_max),
        })
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        let p_center = self.render_from_object * Point3f::ZERO;
        let p_origin = ctx.offset_ray_origin(p_center.into());
        if p_origin.distance_squared(p_center) <= sqr(self.radius) {
            let ray = ctx.spawn_ray(wi);
            let Some(isect) = self.intersect(ray, Float::INFINITY) else { return 0.0 };

            let pdf = (1.0 / self.area()) / isect.intr.interaction.n.dot(-Normal3f::from(wi)).abs() 
                / ctx.p().distance_squared(isect.intr.position());
            if pdf.is_infinite() {
                return 0.0;
            } else {
                return pdf;
            }
        }

        let sin2_theta_max = self.radius * self.radius / ctx.p().distance_squared(p_center);
        let cos_theta_max = safe::sqrt(1.0 - sin2_theta_max);
        let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;
        if sin2_theta_max < 0.00068523 /* sin^2(1.5 deg) */ {
            one_minus_cos_theta_max = sin2_theta_max / 2.0;
        }

        1.0 / (TAU * one_minus_cos_theta_max)
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape::{sphere::Sphere, AbstractShape as _}, transform::Transform, Float, Point3f, Ray, Vec3f};

    #[test]
    fn test_sphere_intersect() {
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            1.0,
            -0.5,
            0.5,
            360.0,
        );
        let ray = Ray::new(Point3f::new(0.0, -2.0, 0.0), Vec3f::new(0.0, 1.0, 0.0));
        assert!(sphere.intersect_predicate(ray, Float::INFINITY));

        let ray = Ray::new(ray.origin, -ray.direction);
        assert!(!sphere.intersect_predicate(ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 0.0, 0.5001), Vec3f::new(0.0, 1.0, 0.0));
        assert!(!sphere.intersect_predicate(ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 0.0, -0.5001), Vec3f::new(0.0, 1.0, 0.0));
        assert!(!sphere.intersect_predicate(ray, Float::INFINITY));
    }
}
