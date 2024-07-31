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
            for i in 1..=n {
                cdf[i] = i as Float / n as Float;
            }
        } else {
            for i in 1..=n {
                cdf[i] /= f_int;
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

        for v in 0..nv {
            marginal_func.push(conditional_v[v].integral());
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
