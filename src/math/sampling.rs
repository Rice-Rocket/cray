use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use vec2d::Vec2D;

use crate::numeric::DifferenceOfProducts;
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

        for i in 1..=n {
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
            conditional_v.push(PiecewiseConstant1D::new_bounded(&f[v * nu..(v * nu) + nu], domain.min.x, domain.max.x));
        }

        let mut marginal_func = Vec::with_capacity(nv);

        for v in conditional_v.iter().take(nv) {
            marginal_func.push(v.integral());
        }

        let marginal = PiecewiseConstant1D::new_bounded(&marginal_func, domain.min.y, domain.max.y);

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

        let iu = ((p.x * self.conditional_v[0].size() as Float) as usize).clamp(0, self.conditional_v[0].size() - 1);
        let iv = ((p.y * self.marginal.size() as Float) as usize).clamp(0, self.marginal.size() - 1);
        self.conditional_v[iv].f[iu] / self.marginal.integral()
    }
}

#[derive(Debug, Clone)]
pub struct SummedAreaTable {
    sum: Vec2D<Float>,
}

impl SummedAreaTable {
    pub fn new(values: &Vec2D<Float>) -> SummedAreaTable {
        let mut sum = Vec2D::from_bounds(values.extent());
        *sum.get_xy_mut(0, 0) = values.get_xy(0, 0);

        for x in 1..sum.width() {
            *sum.get_xy_mut(x, 0) = values.get_xy(x, 0) + sum.get_xy(x - 1, 0);
        }
        for y in 1..sum.height() {
            *sum.get_xy_mut(0, y) = values.get_xy(0, y) + sum.get_xy(0, y - 1);
        }

        for y in 1..sum.height() {
            for x in 1..sum.width() {
                *sum.get_xy_mut(x, y) = values.get_xy(x, y) 
                    + sum.get_xy(x - 1, y) + sum.get_xy(x, y - 1) - sum.get_xy(x - 1, y - 1);
            }
        }

        SummedAreaTable { sum }
    }

    pub fn integral(&self, extent: Bounds2f) -> Float {
        let s = self.lookup(extent.max.x, extent.max.y)
            - self.lookup(extent.min.x, extent.max.y)
            + self.lookup(extent.min.x, extent.min.y)
            - self.lookup(extent.max.x, extent.min.y);
        Float::max(s / (self.sum.width() as Float * self.sum.height() as Float), 0.0)
    }

    fn lookup(&self, mut x: Float, mut y: Float) -> Float {
        x *= self.sum.width() as Float;
        y *= self.sum.height() as Float;
        let x0 = x as i32;
        let y0 = y as i32;

        let v00 = self.lookup_int(x0, y0);
        let v10 = self.lookup_int(x0 + 1, y0);
        let v01 = self.lookup_int(x0, y0 + 1);
        let v11 = self.lookup_int(x0 + 1, y0 + 1);
        let dx = x - x.floor();
        let dy = y - y.floor();

        (1.0 - dx) * (1.0 - dy) * v00
            + (1.0 - dx) * dy * v01
            + dx * (1.0 - dy) * v10
            + dx * dy * v11
    }

    fn lookup_int(&self, mut x: i32, mut y: i32) -> Float {
        if x == 0 || y == 0 {
            return 0.0;
        }

        x = (x - 1).min(self.sum.width() - 1);
        y = (y - 1).min(self.sum.height() - 1);
        self.sum.get_xy(x, y)
    }
}

#[derive(Debug, Clone)]
pub struct WindowedPiecewiseConstant2D {
    sat: SummedAreaTable,
    f: Vec2D<Float>,
}

impl WindowedPiecewiseConstant2D {
    pub fn new(f: Vec2D<Float>) -> WindowedPiecewiseConstant2D {
        WindowedPiecewiseConstant2D {
            sat: SummedAreaTable::new(&f),
            f,
        }
    }

    pub fn sample(&self, u: Point2f, b: Bounds2f, pdf: &mut Float) -> Option<Point2f> {
        let b_int = self.sat.integral(b);

        if b_int == 0.0 {
            return None;
        }

        let px = |x: Float| -> Float {
            let mut bx = b;
            bx.max.x = x;
            self.sat.integral(bx) / b_int
        };

        let mut p = Point2f::ZERO;
        p.x = Self::sample_bisection(px, u[0], b.min.x, b.max.x, self.f.width());

        let nx = self.f.width() as Float;
        let mut b_cond = Bounds2f::new(
            Point2f::new((p.x * nx).floor() / nx, b.min.y),
            Point2f::new((p.x * nx).ceil() / nx, b.max.y),
        );

        if b_cond.min.x == b_cond.max.x {
            b_cond.max.x += 1.0 / nx;
        }

        let cond_integral = self.sat.integral(b_cond);

        if cond_integral == 0.0 {
            return None;
        }

        let py = |y: Float| -> Float {
            let mut by = b_cond;
            by.max.y = y;
            self.sat.integral(by) / cond_integral
        };

        p.y = Self::sample_bisection(py, u[1], b.min.y, b.max.y, self.f.height());

        *pdf = self.eval(p) / b_int;
        Some(p)
    }

    pub fn pdf(&self, p: Point2f, b: Bounds2f) -> Float {
        let f_int = self.sat.integral(b);
        if f_int == 0.0 {
            return 0.0;
        }

        self.eval(p) / f_int
    }

    fn sample_bisection<F>(p: F, u: Float, mut min: Float, mut max: Float, n: i32) -> Float
    where
        F: Fn(Float) -> Float
    {
        while (n as Float * max).ceil() - (n as Float * min).floor() > 1.0 {
            debug_assert!(p(min) <= u);
            debug_assert!(p(max) >= u);
            let mid = (min + max) / 2.0;
            
            if p(mid) > u {
                max = mid;
            } else {
                min = mid;
            }
        }

        let t = (u - p(min)) / (p(max) - p(min));
        Float::clamp(lerp(min, max, t), min, max)
    }

    fn eval(&self, p: Point2f) -> Float {
        let pi = Point2i::new(
            ((p.x * self.f.width() as Float) as i32).min(self.f.width() - 1),
            ((p.y * self.f.height() as Float) as i32).min(self.f.height() - 1),
        );

        self.f.get(pi)
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

pub struct WeightedReservoirSampler<T> {
    rng: SmallRng,
    weight_sum: Float,
    reservoir_weight: Float,
    reservoir: T,
}

impl<T> WeightedReservoirSampler<T> {
    pub fn new(seed: u64, reservoir: T) -> WeightedReservoirSampler<T> {
        WeightedReservoirSampler::<T> {
            rng: SmallRng::seed_from_u64(seed),
            weight_sum: 0.0,
            reservoir_weight: 0.0,
            reservoir,
        }
    }

    pub fn add_sample(&mut self, sample: T, weight: Float) -> bool {
        self.weight_sum += weight;
        let p = weight / self.weight_sum;
        if self.rng.gen::<Float>() < p {
            self.reservoir = sample;
            self.reservoir_weight = weight;
            return true;
        }
        debug_assert!(self.weight_sum < Float::MAX);
        false
    }

    pub fn add_from_callback<F: Fn() -> T>(&mut self, f: F, weight: Float) -> bool {
        self.weight_sum += weight;
        let p = weight / self.weight_sum;
        if self.rng.gen::<Float>() < p {
            self.reservoir = f();
            self.reservoir_weight = weight;
            return true;
        }
        debug_assert!(self.weight_sum < Float::MAX);
        false
    }

    pub fn has_sample(&self) -> bool {
        self.weight_sum > 0.0
    }

    pub fn get_sample(&self) -> &T {
        &self.reservoir
    }

    pub fn sample_probability(&self) -> Float {
        self.reservoir_weight / self.weight_sum
    }

    pub fn weight_sum(&self) -> Float {
        self.weight_sum
    }

    pub fn reset(&mut self) {
        self.reservoir_weight = 0.0;
        self.weight_sum = 0.0;
    }

    pub fn merge(&mut self, other: &WeightedReservoirSampler<T>)
    where
        T: Clone
    {
        debug_assert!(self.weight_sum + other.weight_sum < Float::MAX);
        if other.has_sample() && self.add_sample(other.reservoir.clone(), other.weight_sum) {
            self.reservoir_weight = other.reservoir_weight;
        }
    }
}

impl<T: Default> Default for WeightedReservoirSampler<T> {
    fn default() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(0),
            weight_sum: 0.0,
            reservoir_weight: 0.0,
            reservoir: T::default(),
        }
    }
}

pub fn power_heuristic(nf: u8, f_pdf: Float, ng: u8, g_pdf: Float) -> Float {
    let f = nf as Float * f_pdf;
    let g = ng as Float * g_pdf;
    if sqr(f).is_infinite() {
        1.0
    } else {
        (f * f) / (f * f + g * g)
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

    r * Point2f::new(theta.cos(), theta.sin())
}

pub fn sample_uniform_disk_polar(u: Point2f) -> Point2f {
    let r = Float::sqrt(u.x);
    let theta = 2.0 * PI * u.y;
    Point2f::new(r * Float::cos(theta), r * Float::sin(theta))
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
        *u_remapped = Float::min((up - sum) / weights[offset], ONE_MINUS_EPSILON);
    }

    Some(offset)
}

pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    debug_assert!(a >= 0.0 && b >= 0.0);
    if u == 0.0 && a == 0.0 {
        return 0.0;
    }

    let x = u * (a + b) / (a + Float::sqrt(lerp(a * a, b * b, u)));
    Float::min(x, ONE_MINUS_EPSILON)
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

pub fn sample_uniform_triangle(u: Point2f) -> (Float, Float, Float) {
    let (b0, b1) = if u[0] < u[1] {
        let b0 = u[0] / 2.0;
        let b1 = u[1] - b0;
        (b0, b1)
    } else {
        let b1 = u[1] / 2.0;
        let b0 = u[0] - b1;
        (b0, b1)
    };
    (b0, b1, 1.0 - b1 - b0)
}

pub fn sample_bilinear(u: Point2f, w: &[Float]) -> Point2f {
    debug_assert_eq!(4, w.len());
    let y = sample_linear(u[1], w[0] + w[1], w[2] + w[3]);
    let x = sample_linear(u[0], lerp(w[0], w[2], y), lerp(w[1], w[3], y));
    Point2f { x, y }
}

pub fn invert_bilinear(p: Point2f, vert: &[Point2f]) -> Point2f {
    let a = vert[0];
    let b = vert[1];
    let c = vert[2];
    let d = vert[3];
    let e = b - a;
    let f = d - a;
    let g = (a - b) + (c - d);
    let h = p - a;

    let orthogonal = |a: Point2f, b: Point2f| {
        Float::difference_of_products(a.x, b.y, a.y, b.x)
    };

    let k2 = orthogonal(g, f);
    let k1 = orthogonal(e, f) + orthogonal(h, g);
    let k0 = orthogonal(h, e);

    if Float::abs(k2) < 0.001 {
        if Float::abs(e.x * k1 - g.x * k0) < 1e-5 {
            return Point2f::new((h.y * k1 + f.y * k0) / (e.y * k1 - g.y * k0), -k0 / k1);
        } else {
            return Point2f::new((h.x * k1 + f.x * k0) / (e.x * k1 - g.x * k0), -k0 / k1);
        }
    }

    let Some((v0, v1)) = quadratic(k2, k1, k0) else {
        return Point2f::ZERO;
    };

    let u = (h.x - f.x * v0) / (e.x + g.x * v0);
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v0) {
        Point2f::new((h.x - f.x * v1) / (e.x + g.x * v1), v1)
    } else {
        Point2f::new(u, v0)
    }
}

pub fn invert_spherical_rectangle_sample(
    p_ref: Point3f,
    s: Point3f,
    ex: Vec3f,
    ey: Vec3f,
    p_rect: Point3f,
) -> Point2f {
    let exl = ex.length();
    let eyl = ey.length();
    let mut r = Frame::from_xy(ex / exl, ey / eyl);

    let d = s - p_ref;
    let d_local = r.localize(d.into());
    let mut z0 = d_local.z;

    if z0 > 0.0 {
        r.z = -r.z;
        z0 *= -1.0;
    }
    let z0sq = sqr(z0);
    let x0 = d_local.x;
    let y0 = d_local.y;
    let x1 = x0 + exl;
    let y1 = y0 + eyl;
    let y0sq = sqr(y0);
    let y1sq = sqr(y1);

    let v00 = Vec3f::new(x0, y0, z0);
    let v01 = Vec3f::new(x0, y1, z0);
    let v10 = Vec3f::new(x1, y0, z0);
    let v11 = Vec3f::new(x1, y1, z0);

    let n0 = v00.cross(v10).normalize();
    let n1 = v10.cross(v11).normalize();
    let n2 = v11.cross(v01).normalize();
    let n3 = v01.cross(v00).normalize();

    let g0 = (-n0).angle_between(n1);
    let g1 = (-n1).angle_between(n2);
    let g2 = (-n2).angle_between(n3);
    let g3 = (-n3).angle_between(n0);

    let b0 = n0.z;
    let b1 = n2.z;
    let b0sq = sqr(b0);
    let b1sq = sqr(b1);

    let solid_angle = g0 + g1 + g2 + g3 - 2.0 * PI;

    // TODO: this (rarely) goes differently than sample. figure out why...
    if solid_angle < 1e-3 {
        let pq = p_rect - s;
        return Point2f::new(pq.dot(ex) / ex.length_squared(), pq.dot(ey) / ey.length_squared());
    }

    let v = r.localize((p_rect - p_ref).into());
    let mut xu = v.x;
    let yv = v.y;

    xu = Float::clamp(xu, x0, x1);
    if xu == 0.0 {
        xu = 1e-10;
    }

    let invcusq = 1.0 + z0sq / sqr(xu);
    let fusq = invcusq - b0sq;
    let fu = Float::copysign(Float::sqrt(fusq), xu);

    let sqrt = safe::sqrt(Float::difference_of_products(b0, b0, b1, b1) + fusq);
    let mut au = Float::atan2(
        -(b1 * fu) - Float::copysign(b0 * sqrt, fu * b0),
        b0 * b1 - sqrt * Float::abs(fu),
    );
    if au > 0.0 {
        au -= 2.0 * PI;
    }

    if fu == 0.0 {
        au = PI;
    }

    let u0 = (au + g2 + g3) / solid_angle;

    let ddsq = sqr(xu) + z0sq;
    let dd = Float::sqrt(ddsq);
    let h0 = y0 / Float::sqrt(ddsq + y0sq);
    let h1 = y1 / Float::sqrt(ddsq + y1sq);
    let yvsq = sqr(yv);

    let u1  = [
        (Float::difference_of_products(h0, h0, h0, h1) - Float::abs(h0 - h1) 
            * Float::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) / sqr(h0 - h1),
        (Float::difference_of_products(h0, h0, h0, h1) + Float::abs(h0 - h1) 
            * Float::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) / sqr(h0 - h1),
    ];

    // TODO: yuck is there a better way to figure out which is the right
    // solution?
    let hv = [lerp(h0, h1, u1[0]), lerp(h0, h1, u1[1])];
    let hvsq = [sqr(hv[0]), sqr(hv[1])];
    let yz = [
        (hv[0] * dd) / Float::sqrt(1.0 - hvsq[0]),
        (hv[1] * dd) / Float::sqrt(1.0 - hvsq[1]),
    ];

    if Float::abs(yz[0] - yv) < Float::abs(yz[1] - yv) {
        Point2f::new(Float::clamp(u0, 0.0, 1.0), u1[0])
    } else {
        Point2f::new(Float::clamp(u0, 0.0, 1.0), u1[1])
    }
}

pub fn bilinear_pdf(p: Point2f, w: &[Float]) -> Float {
    debug_assert_eq!(4, w.len());
    if p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 {
        return 0.0;
    }
    if w[0] + w[1] + w[2] + w[3] == 0.0 {
        return 1.0;
    }
    4.0 * ((1.0 - p[0]) * (1.0 - p[1]) * w[0]
        + p[0] * (1.0 - p[1]) * w[1]
        + (1.0 - p[0]) * p[1] * w[2]
        + p[0] * p[1] * w[3])
        / (w[0] + w[1] + w[2] + w[3])
}

pub fn sample_spherical_triangle(v: &[Point3f; 3], p: Point3f, u: Point2f) -> ([Float; 3], Float) {
    let a = v[0] - p;
    let b = v[1] - p;
    let c = v[2] - p;
    debug_assert!(a.length_squared() > 0.0);
    debug_assert!(b.length_squared() > 0.0);
    debug_assert!(c.length_squared() > 0.0);

    let a = a.normalize();
    let b = b.normalize();
    let c = c.normalize();

    let n_ab = a.cross(b);
    let n_bc = b.cross(c);
    let n_ca = c.cross(a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        // TODO: Consider using an Option return type instead.
        return ([0.0, 0.0, 0.0], 0.0);
    }

    let n_ab = n_ab.normalize();
    let n_bc = n_bc.normalize();
    let n_ca = n_ca.normalize();

    let alpha = n_ab.angle_between(-n_ca);
    let beta = n_bc.angle_between(-n_ab);
    let gamma = n_ca.angle_between(-n_bc);

    let a_pi = alpha + beta + gamma;
    let ap_pi = lerp(PI, a_pi, u[0]);
    let area = a_pi - PI;
    let pdf = if area <= 0.0 { 0.0 } else { 1.0 / area };

    let cos_alpha = Float::cos(alpha);
    let sin_alpha = Float::sin(alpha);
    let sin_phi = Float::sin(ap_pi) * cos_alpha - Float::cos(ap_pi) * sin_alpha;
    let cos_phi = Float::cos(ap_pi) * cos_alpha + Float::sin(ap_pi) * sin_alpha;
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * a.dot(b);
    let cos_bp = (k2 + (Float::difference_of_products(k2, cos_phi, k1, sin_phi)) * cos_alpha)
        / (Float::sum_of_products(k2, sin_phi, k1, cos_phi) * sin_alpha);

    debug_assert!(!cos_bp.is_nan());
    let cos_bp = cos_bp.clamp(-1.0, 1.0);

    let sin_bp = safe::sqrt(1.0 - cos_bp * cos_bp);
    let cp = cos_bp * a + sin_bp * c.gram_schmidt(a).normalize();

    let cos_theta = 1.0 - u[1] * (1.0 - cp.dot(b));
    let sin_theta = safe::sqrt(1.0 - cos_theta * cos_theta);
    let w = cos_theta * b + sin_theta * cp.gram_schmidt(b).normalize();

    let e1 = v[1] - v[0];
    let e2 = v[2] - v[0];
    let s1 = w.cross(e2);
    let divisor = e1.dot(e1);

    if divisor == 0.0 {
        return ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], pdf);
    }

    let inv_divisor = 1.0 / divisor;
    let s = p - v[0];
    let b1 = s.dot(s1) * inv_divisor;
    let b2 = w.dot(s.cross(e1)) * inv_divisor;

    let b1 = b1.clamp(0.0, 1.0);
    let b2 = b2.clamp(0.0, 1.0);
    let (b1, b2) = if b1 + b2 > 1.0 {
        (b1 / (b1 + b2), b2 / (b1 + b2))
    } else {
        (b1, b2)
    };
    ([1.0 - b1 - b2, b1, b2], pdf)
}

pub fn invert_spherical_triangle_sample(v: &[Point3f; 3], p: Point3f, w: Vec3f) -> Point2f {
    let a = v[0] - p;
    let b = v[1] - p;
    let c = v[2] - p;
    debug_assert!(a.length_squared() > 0.0);
    debug_assert!(b.length_squared() > 0.0);
    debug_assert!(c.length_squared() > 0.0);

    let a = a.normalize();
    let b = b.normalize();
    let c = c.normalize();

    let n_ab = a.cross(b);
    let n_bc = b.cross(c);
    let n_ca = c.cross(a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        // TODO: Consider using an Option return type instead.
        return Point2f::ZERO;
    }

    let n_ab = n_ab.normalize();
    let n_bc = n_bc.normalize();
    let n_ca = n_ca.normalize();

    let alpha = n_ab.angle_between(-n_ca);
    let beta = n_bc.angle_between(-n_ab);
    let gamma = n_ca.angle_between(-n_bc);

    let cp = b.cross(w.into()).cross(c.cross(a)).normalize();
    let cp = if cp.dot(a + c) < 0.0 { -cp } else { cp };

    let u0 = if a.dot(cp) > 0.99999847691 {
        0.0
    } else {
        let n_cpb = cp.cross(b);
        let n_acp = a.cross(cp);
        if n_cpb.length_squared() == 0.0 || n_acp.length_squared() == 0.0 {
            return Point2f::new(0.5, 0.5);
        }
        let n_cpb = n_cpb.normalize();
        let n_acp = n_acp.normalize();
        let ap = alpha + n_ab.angle_between(n_cpb) + n_acp.angle_between(-n_cpb) - PI;

        let area = alpha + beta + gamma - PI;
        ap / area
    };

    let u1 = (1.0 - w.dot(b)) / (1.0 - cp.dot(b));
    Point2f::new(u0.clamp(0.0, 1.0), u1.clamp(0.0, 1.0))
}

pub fn sample_spherical_rectangle(
    p_ref: Point3f,
    s: Point3f,
    ex: Vec3f,
    ey: Vec3f,
    u: Point2f,
    pdf: Option<&mut Float>,
) -> Point3f {
    let exl = ex.length();
    let eyl = ey.length();
    let mut r = Frame::from_xy(ex / exl, ey / eyl);
    let d_local = r.localize((s - p_ref).into());
    let mut z0 = d_local.z;
    
    if z0 > 0.0 {
        r.z = -r.z;
        z0 *= -1.0;
    }

    let x0 = d_local.x;
    let y0 = d_local.y;
    let x1 = x0 + exl;
    let y1 = y0 + eyl;

    let v00 = Vec3f::new(x0, y0, z0);
    let v01 = Vec3f::new(x0, y1, z0);
    let v10 = Vec3f::new(x1, y0, z0);
    let v11 = Vec3f::new(x1, y1, z0);
    let n0 = v00.cross(v10).normalize();
    let n1 = v10.cross(v11).normalize();
    let n2 = v11.cross(v01).normalize();
    let n3 = v01.cross(v00).normalize();

    let g0 = (-n0).angle_between(n1);
    let g1 = (-n1).angle_between(n2);
    let g2 = (-n2).angle_between(n3);
    let g3 = (-n3).angle_between(n0);

    let solid_angle = g0 + g1 + g2 + g3 - 2.0 * PI;
    if solid_angle <= 0.0 {
        if let Some(pdf) = pdf {
            *pdf = 0.0;
        }
        return Point3f::from(s + u[0] * ex + u[1] * ey);
    }
    if let Some(pdf) = pdf {
        *pdf = Float::max(0.0, 1.0 / solid_angle);
    }
    if solid_angle < 1e-3 {
        return Point3f::from(s + u[0] * ex + u[1] * ey);
    }

    let b0 = n0.z;
    let b1 = n2.z;
    let au = u[0] * (g0 + g1 - 2.0 * PI) + (u[0] - 1.0) * (g2 + g3);
    let fu = (Float::cos(au) * b0 - b1) / Float::sin(au);
    let cu = Float::copysign(1.0 / Float::sqrt(sqr(fu) + sqr(b0)), fu);
    let cu = Float::clamp(cu, -ONE_MINUS_EPSILON, ONE_MINUS_EPSILON);

    let xu = -(cu * z0) / safe::sqrt(1.0 - sqr(cu));
    let xu = Float::clamp(xu, x0, x1);

    let dd = Float::sqrt(sqr(xu) + sqr(z0));
    let h0 = y0 / Float::sqrt(sqr(dd) + sqr(y0));
    let h1 = y1 / Float::sqrt(sqr(dd) + sqr(y1));
    let hv = h0 + u[1] * (h1 - h0);
    let hvsq = sqr(hv);
    let yv = if hvsq < 1.0 - 1e-6 {
        (hv * dd) / Float::sqrt(1.0 - hvsq)
    } else {
        y1
    };

    p_ref + r.from_local(Vec3f::new(xu, yv, z0))
}

pub fn sample_exponential(x: Float, a: Float) -> Float {
    a * Float::exp(-a * x)
}
