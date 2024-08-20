use vec2d::Vec2D;

use crate::math::*;

#[derive(Debug, Clone, Default)]
pub struct PiecewiseConstant1D {
    f: Vec<Float>,
    cdf: Vec<Float>,
    min: Float,
    max: Float,
    f_int: Float,
}

impl PiecewiseConstant1D {
    pub fn new(f: &[Float]) -> PiecewiseConstant1D {
        Self::new_bounded(f, 0.0, 1.0)
    }

    pub fn new_bounded(f: &[Float], min: Float, max: Float) -> PiecewiseConstant1D {
        assert!(max > min);

        let func: Vec<_> = f.iter().map(|v| v.abs()).collect();
        let n = func.len();
        let mut cdf = vec![0.0; n + 1];
        cdf[0] = 0.0;

        for i in 0..=n {
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n as Float;
        }

        let f_int = cdf[n];
        if f_int == 0.0 {
            for (i, c) in cdf.iter_mut().enumerate().take(n + 1).skip(1) {
                *c = i as Float / n as Float;
            }
        } else {
            for c in cdf.iter_mut().take(n + 1).skip(1) {
                *c /= f_int;
            }
        }

        PiecewiseConstant1D {
            f: func,
            cdf,
            min,
            max,
            f_int,
        }
    }

    pub fn integral(&self) -> Float { self.f_int }

    pub fn size(&self) -> usize { self.f.len() }

    /// Returns (value, pdf, offset)
    pub fn sample(&self, u: Float) -> (Float, Float, usize) {
        let offset = find_interval(self.cdf.len(), |i| self.cdf[i] <= u);

        let mut du = u - self.cdf[offset];
        if self.cdf[offset + 1] - self.cdf[offset] > 0.0 {
            du /= self.cdf[offset + 1] - self.cdf[offset]
        }

        let pdf = if self.f_int > 0.0 {
            self.f[offset] / self.f_int
        } else {
            0.0
        };

        let value = lerp(self.min, self.max, (offset as Float + du) / self.size() as Float);

        (value, pdf, offset)
    }

    pub fn invert(&self, x: Float) -> Option<Float> {
        if x < self.min || x > self.max {
            return None;
        }

        let c = (x - self.min) / (self.max - self.min) * self.f.len() as Float;
        let offset = (c as usize).clamp(0, self.f.len() - 1);
        debug_assert!(offset + 1 < self.cdf.len());

        let delta = c - offset as Float;
        Some(lerp(self.cdf[offset], self.cdf[offset + 1], delta))
    }
}

#[derive(Debug, Clone, Default)]
pub struct PiecewiseConstant2D {
    domain: Bounds2f,
    conditional_v: Vec<PiecewiseConstant1D>,
    marginal: PiecewiseConstant1D,
}

impl PiecewiseConstant2D {
    pub fn new_from_2d(data: &Vec2D<Float>, domain: Bounds2f) -> PiecewiseConstant2D {
        PiecewiseConstant2D::new(&data.data, data.width() as usize, data.height() as usize, domain)
    }

    pub fn new(f: &[Float], nu: usize, nv: usize, domain: Bounds2f) -> PiecewiseConstant2D {
        assert_eq!(f.len(), nu * nv);
        let mut conditional_v = Vec::with_capacity(nv);

        for v in 0..nv {
            conditional_v.push(PiecewiseConstant1D::new_bounded(&f[v * nu..(v * nu) + nu], domain.min[0], domain.max[0]));
        }

        let mut marginal_func = Vec::with_capacity(nv);

        for v in conditional_v.iter().take(nv) {
            marginal_func.push(v.integral());
        }

        let marginal = PiecewiseConstant1D::new_bounded(&marginal_func, domain.min[1], domain.max[1]);

        PiecewiseConstant2D {
            domain,
            conditional_v,
            marginal,
        }
    }

    pub fn integral(&self) -> Float { self.marginal.integral() }

    pub fn domain(&self) -> &Bounds2f { &self.domain }

    pub fn resolution(&self) -> Point2i { Point2i::new(self.conditional_v[0].size() as i32, self.marginal.size() as i32) }

    /// Returns (sampled value, PDF, offset)
    pub fn sample(&self, u: Point2f) -> (Point2f, Float, Point2i) {
        let (d1, pdf1, uv1) = self.marginal.sample(u[1]);
        let (d0, pdf0, uv0) = self.conditional_v[uv1].sample(u[0]);
        let value = Point2f::new(d0, d1);
        let pdf = pdf0 * pdf1;
        let offset = Point2i::new(uv0 as i32, uv1 as i32);
        (value, pdf, offset)
    }

    pub fn pdf(&self, pr: Point2f) -> Float {
        let p = self.domain.offset(pr);

        let iu = ((p[0] * self.conditional_v[0].size() as Float) as usize).clamp(0, self.conditional_v[0].size() - 1);
        let iv = ((p[1] * self.marginal.size() as Float) as usize).clamp(0, self.marginal.size() - 1);
        self.conditional_v[iv].f[iu] / self.marginal.integral()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampledGrid<T> {
    values: Vec<T>,
    n: Vec3i,
}

impl<T> SampledGrid<T> {
    pub fn new(v: Vec<T>, nx: i32, ny: i32, nz: i32) -> SampledGrid<T> {
        SampledGrid::<T> {
            values: v,
            n: Vec3i::new(nx, ny, nz),
        }
    }

    pub fn x_size(&self) -> i32 { self.n.x }
    pub fn y_size(&self) -> i32 { self.n.y }
    pub fn z_size(&self) -> i32 { self.n.z }

    pub fn lookup_convert<F, U>(&self, p: Point3f, convert: F) -> Option<U>
    where
        F: Fn(&T) -> U,
        U: Add<U, Output = U> + Mul<Float, Output = U> + Copy,
    {
        let p_samples = Point3f::new(
            p.x * self.n.x as Float - 0.5,
            p.y * self.n.y as Float - 0.5,
            p.z * self.n.z as Float - 0.5,
        );

        let pi = Point3i::new(p_samples.x.floor() as i32, p_samples.y.floor() as i32, p_samples.z.floor() as i32);
        let d = p_samples - Point3f::new(pi.x as Float, pi.y as Float, pi.z as Float);

        // TODO: There could be an issue here for edges
        let d00 = lerp_float(
            self.lookup_aligned_convert(pi, &convert)?,
            self.lookup_aligned_convert(pi + Vec3i::new(1, 0, 0), &convert)?,
            d.x);
        let d10 = lerp_float(
            self.lookup_aligned_convert(pi + Vec3i::new(0, 1, 0), &convert)?,
            self.lookup_aligned_convert(pi + Vec3i::new(1, 1, 0), &convert)?,
            d.x);
        let d01 = lerp_float(
            self.lookup_aligned_convert(pi + Vec3i::new(0, 0, 1), &convert)?,
            self.lookup_aligned_convert(pi + Vec3i::new(1, 0, 1), &convert)?,
            d.x);
        let d11 = lerp_float(
            self.lookup_aligned_convert(pi + Vec3i::new(0, 1, 1), &convert)?,
            self.lookup_aligned_convert(pi + Vec3i::new(1, 1, 1), &convert)?,
            d.x);

        Some(lerp_float(lerp_float(d00, d10, d.y), lerp_float(d01, d11, d.y), d.z))
    }

    pub fn lookup(&self, p: Point3f) -> Option<T>
    where
        T: Add<T, Output = T> + Mul<Float, Output = T> + Copy,
    {
        let p_samples = Point3f::new(
            p.x * self.n.x as Float - 0.5,
            p.y * self.n.y as Float - 0.5,
            p.z * self.n.z as Float - 0.5,
        );

        let pi = Point3i::new(p_samples.x.floor() as i32, p_samples.y.floor() as i32, p_samples.z.floor() as i32);
        let d = p_samples - Point3f::new(pi.x as Float, pi.y as Float, pi.z as Float);

        let d00 = lerp_float(
            self.lookup_aligned(pi)?,
            self.lookup_aligned(pi + Vec3i::new(1, 0, 0))?,
            d.x);
        let d10 = lerp_float(
            self.lookup_aligned(pi + Vec3i::new(0, 1, 0))?,
            self.lookup_aligned(pi + Vec3i::new(1, 1, 0))?,
            d.x);
        let d01 = lerp_float(
            self.lookup_aligned(pi + Vec3i::new(0, 0, 1))?,
            self.lookup_aligned(pi + Vec3i::new(1, 0, 1))?,
            d.x);
        let d11 = lerp_float(
            self.lookup_aligned(pi + Vec3i::new(0, 1, 1))?,
            self.lookup_aligned(pi + Vec3i::new(1, 1, 1))?,
            d.x);

        Some(lerp_float(lerp_float(d00, d10, d.y), lerp_float(d01, d11, d.y), d.z))
    }

    pub fn lookup_aligned_convert<F, U>(&self, p: Point3i, convert: F) -> Option<U>
    where
        F: Fn(&T) -> U,
    {
        let sample_bounds = Bounds3i::new(Point3i::ZERO, Point3i::new(self.n.x, self.n.y, self.n.z));
        if !sample_bounds.inside_exclusive(p) {
            return None;
        }

        Some(convert(&self.values[((p.z * self.n.y + p.y) * self.n.x + p.x) as usize]))
    }

    pub fn lookup_aligned(&self, p: Point3i) -> Option<T>
    where
        T: Copy
    {
        let sample_bounds = Bounds3i::new(Point3i::ZERO, Point3i::new(self.n.x, self.n.y, self.n.z));
        if !sample_bounds.inside_exclusive(p) {
            return None;
        }

        Some(self.values[((p.z * self.n.y + p.y) * self.n.x + p.x) as usize])
    }

    pub fn max_value_convert<F>(&self, bounds: Bounds3f, convert: F) -> Float
    where
        F: Fn(&T) -> Float,
    {
        let ps = [
            Point3f::new(
                bounds.min.x * self.n.x as Float - 0.5,
                bounds.min.y * self.n.y as Float - 0.5,
                bounds.min.z * self.n.z as Float - 0.5,
            ).floor(),
            Point3f::new(
                bounds.max.x * self.n.x as Float - 0.5,
                bounds.max.y * self.n.y as Float - 0.5,
                bounds.max.z * self.n.z as Float - 0.5,
            ).floor(),
        ];

        let pi = [
            Point3i::new(ps[0].x as i32, ps[0].y as i32, ps[0].z as i32).max(Point3i::ZERO),
            (Point3i::new(ps[1].x as i32, ps[1].y as i32, ps[1].z as i32) + Vec3i::ONE)
                .min(Point3i::new(self.n.x - 1, self.n.y - 1, self.n.z - 1))
        ];

        let mut max_value = self.lookup_aligned_convert(pi[0], &convert).unwrap_or(Float::MIN);
        for z in pi[0].z..=pi[1].z {
            for y in pi[0].y..=pi[1].y {
                for x in pi[0].x..=pi[1].x {
                    max_value = max_value.max(self.lookup_aligned_convert(Point3i::new(x, y, z), &convert).unwrap_or(Float::MIN));
                }
            }
        }

        max_value
    }

    pub fn max_value(&self, bounds: Bounds3f) -> T
    where
        T: num::Float,
    {
        let ps = [
            Point3f::new(
                bounds.min.x * self.n.x as Float - 0.5,
                bounds.min.y * self.n.y as Float - 0.5,
                bounds.min.z * self.n.z as Float - 0.5,
            ).floor(),
            Point3f::new(
                bounds.max.x * self.n.x as Float - 0.5,
                bounds.max.y * self.n.y as Float - 0.5,
                bounds.max.z * self.n.z as Float - 0.5,
            ).floor(),
        ];

        let pi = [
            Point3i::new(ps[0].x as i32, ps[0].y as i32, ps[0].z as i32).max(Point3i::ZERO),
            (Point3i::new(ps[1].x as i32, ps[1].y as i32, ps[1].z as i32) + Vec3i::ONE)
                .min(Point3i::new(self.n.x - 1, self.n.y - 1, self.n.z - 1))
        ];

        let mut max_value = self.lookup_aligned(pi[0]).unwrap_or(T::min_value());
        for z in pi[0].z..=pi[1].z {
            for y in pi[0].y..=pi[1].y {
                for x in pi[0].x..=pi[1].x {
                    max_value = max_value.max(self.lookup_aligned(Point3i::new(x, y, z)).unwrap_or(T::min_value()));
                }
            }
        }

        max_value
    }
}

pub fn sample_uniform_disk_concentric(u: Point2f) -> Point2f {
    let u_offset = u * 2.0 - Vec2f::new(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::ZERO;
    }

    let (theta, r) = if u_offset.x.abs() > u_offset.y.abs() {
        (FRAC_PI_4 * (u_offset.y / u_offset.x), u_offset.x)
    } else {
        (FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y), u_offset.y)
    };

    Point2f::new(theta.cos(), theta.sin()) * r
}

pub fn sample_uniform_sphere(u: Point2f) -> Vec3f {
    let z = 1.0 - 2.0 * u.x;
    let r = safe::sqrt(1.0 - z * z);
    let phi = TAU * u.y;

    Vec3f::new(r * phi.cos(), r * phi.sin(), z)
}

#[inline(always)]
pub fn uniform_sphere_pdf() -> Float {
    FRAC_1_4PI
}

pub fn sample_discrete(
    weights: &[Float],
    u: Float,
    pmf: Option<&mut Float>,
    u_remapped: Option<&mut Float>,
) -> Option<usize> {
    if weights.is_empty() {
        if let Some(pmf) = pmf {
            *pmf = 0.0;
        }
        return None;
    }

    let sum_weights: Float = weights.iter().sum();

    let up = u * sum_weights;
    let up = if up == sum_weights {
        next_float_down(up)
    } else {
        up
    };

    let mut offset = 0;
    let mut sum: Float = 0.0;
    while sum + weights[offset] <= up {
        sum += weights[offset];
        offset += 1;
        debug_assert!(offset < weights.len());
    }

    if let Some(pmf) = pmf {
        *pmf = weights[offset] / sum_weights;
    }
    if let Some(u_remapped) = u_remapped {
        *u_remapped = Float::min((up - sum) / weights[offset], 1.0 - Float::EPSILON);
    }

    Some(offset)
}

pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    debug_assert!(a >= 0.0 && b >= 0.0);
    if u == 0.0 && a == 0.0 {
        return 0.0;
    }

    let x = u * (a + b) / (a + Float::sqrt(lerp(a * a, b * b, u)));
    Float::min(x, 1.0 - Float::EPSILON)
}

pub fn sample_tent(mut u: Float, r: Float) -> Float {
    if let Some(s) = sample_discrete(&[0.5, 0.5], u, None, Some(&mut u)) {
        if s == 0 {
            return -r + r * sample_linear(u, 0.0, 1.0)
        }
    }

    r * sample_linear(u, 1.0, 0.0)
}

pub fn sample_uniform_hemisphere(u: Point2f) -> Vec3f {
    let z = u.x;
    let r = safe::sqrt(1.0 - z * z);
    let phi = TAU * u.y;

    Vec3f::new(r * Float::cos(phi), r * Float::sin(phi), z)
}

#[inline(always)]
pub fn uniform_hemisphere_pdf() -> Float {
    FRAC_1_4PI
}

pub fn sample_cosine_hemisphere(u: Point2f) -> Vec3f {
    let d = sample_uniform_disk_concentric(u);
    let z = safe::sqrt(1.0 - sqr(d.x) - sqr(d.y));
    Vec3f::new(d.x, d.y, z)
}

pub fn cosine_hemisphere_pdf(cos_theta: Float) -> Float {
    cos_theta * FRAC_1_PI
}
