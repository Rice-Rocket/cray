use sphere::Sphere;

use crate::{interaction::{Interaction, SurfaceInteraction}, Bounds3f, DirectionCone, Float, Normal3f, Point2f, Point3f, Point3fi, Ray, Vec3f};

pub mod sphere;

pub trait AbstractShape {
    /// Spatial extent of the shape
    fn bounds(&self) -> Bounds3f;

    /// Extent of the range of surface normals
    fn normal_bounds(&self) -> DirectionCone;

    /// Finds the first ray-shape intersection from (0, `t_max`), or [`None`] if there is no
    /// intersection. The rays are passed in rendering space, so shapes are responsible for transforming them to
    /// object space if needed for intersection tests. The intersection information returned should
    /// be in rendering space.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Detects if an intersection occurred, but does not return information about the
    /// intersection.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;

    /// Surface area of the shape in rendering space
    fn area(&self) -> Float;

    /// Samples a point on the surface of the shape
    fn sample(&self, u: Point2f) -> Option<ShapeSample>;

    /// Probability density for sampling the specified point on the shape that corresponds to the
    /// given [`Interaction`]. Should only be called with interactions on the surface of the shape.
    fn pdf(&self, interaction: &Interaction) -> Float;

    /// Like sample(), but takes a reference point from which the shape is being viewed. 
    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample>;

    /// Returns the shape's probability for sampling a point on the light such that the incident
    /// direction at the reference point is `wi`. The density should be with respect to the solid
    /// angle at the reference point. 
    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float;
}

#[derive(Debug, Clone)]
pub enum Shape {
    Sphere(Sphere),
}

impl AbstractShape for Shape {
    fn bounds(&self) -> Bounds3f {
        match self {
            Shape::Sphere(s) => s.bounds(),
        }
    }

    fn normal_bounds(&self) -> DirectionCone {
        match self {
            Shape::Sphere(s) => s.normal_bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere(s) => s.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Shape::Sphere(s) => s.intersect_predicate(ray, t_max),
        }
    }

    fn area(&self) -> Float {
        match self {
            Shape::Sphere(s) => s.area(),
        }
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        match self {
            Shape::Sphere(s) => s.sample(u),
        }
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        match self {
            Shape::Sphere(s) => s.pdf(interaction),
        }
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        match self {
            Shape::Sphere(s) => s.sample_with_context(ctx, u),
        }
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vec3f) -> Float {
        match self {
            Shape::Sphere(s) => s.pdf_with_context(ctx, wi),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShapeIntersection {
    pub intr: SurfaceInteraction,
    pub t_hit: Float,
}

pub struct QuadricIntersection {
    pub t_hit: Float,
    pub p_obj: Point3f,
    pub phi: Float,
}

pub struct ShapeSample {
    pub intr: Interaction,
    pub pdf: Float,
}

pub struct ShapeSampleContext {
    pub pi: Point3fi,
    pub n: Normal3f,
    pub ns: Normal3f,
    pub time: Float,
}

impl ShapeSampleContext {
    pub fn new(pi: Point3fi, n: Normal3f, ns: Normal3f, time: Float) -> ShapeSampleContext {
        ShapeSampleContext { pi, n, ns, time }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> ShapeSampleContext {
        ShapeSampleContext {
            pi: si.interaction.pi,
            n: si.interaction.n,
            ns: si.shading.n,
            time: si.interaction.time,
        }
    }

    pub fn p(&self) -> Point3f {
        self.pi.into()
    }

    pub fn offset_ray_origin(&self, w: Vec3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    pub fn offset_ray_origin_pt(&self, pt: Point3f) -> Point3f {
        self.offset_ray_origin((pt - self.p()).into())
    }

    pub fn spawn_ray(&self, w: Vec3f) -> Ray {
        Ray::new_with_time(
            self.offset_ray_origin(w),
            w,
            self.time,
        )
    }
}
