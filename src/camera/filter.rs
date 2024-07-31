use rand::Rng;

use crate::{gaussian, gaussian_integral, lerp, options::Options, reader::{paramdict::ParameterDictionary, target::FileLoc}, sampler::AbstractSampler, sampling::{sample_tent, PiecewiseConstant2D}, sqr, vec2d::Vec2D, windowed_sinc, Bounds2f, Bounds2i, Float, Point2f, Point2i, Vec2f};

use super::CameraSample;

pub trait AbstractFilter {
    fn radius(&self) -> Vec2f;

    fn evaluate(&self, p: Point2f) -> Float;

    fn integral(&self) -> Float;

    fn sample(&self, u: Point2f) -> FilterSample;
}

#[derive(Debug, Clone)]
pub enum Filter {
    Box(BoxFilter),
    Triangle(TriangleFilter),
    Gaussian(GaussianFilter),
    Mitchell(MitchellFilter),
    LanczosSinc(LanczosSincFilter),
}

impl Filter {
    pub fn create(name: &str, parameters: &mut ParameterDictionary, loc: &FileLoc) -> Filter {
        match name {
            "box" => Filter::Box(BoxFilter::create(parameters, loc)),
            "triangle" => Filter::Triangle(TriangleFilter::create(parameters, loc)),
            "gaussian" => Filter::Gaussian(GaussianFilter::create(parameters, loc)),
            "mitchell" => Filter::Mitchell(MitchellFilter::create(parameters, loc)),
            "sinc" => Filter::LanczosSinc(LanczosSincFilter::create(parameters, loc)),
            _ => panic!("unknown filter type")
        }
    }
}

impl AbstractFilter for Filter {
    fn radius(&self) -> Vec2f {
        match self {
            Filter::Box(f) => f.radius(),
            Filter::Triangle(f) => f.radius(),
            Filter::Gaussian(f) => f.radius(),
            Filter::Mitchell(f) => f.radius(),
            Filter::LanczosSinc(f) => f.radius(),
        }
    }

    fn evaluate(&self, p: Point2f) -> Float {
        match self {
            Filter::Box(f) => f.evaluate(p),
            Filter::Triangle(f) => f.evaluate(p),
            Filter::Gaussian(f) => f.evaluate(p),
            Filter::Mitchell(f) => f.evaluate(p),
            Filter::LanczosSinc(f) => f.evaluate(p),
        }
    }

    fn integral(&self) -> Float {
        match self {
            Filter::Box(f) => f.integral(),
            Filter::Triangle(f) => f.integral(),
            Filter::Gaussian(f) => f.integral(),
            Filter::Mitchell(f) => f.integral(),
            Filter::LanczosSinc(f) => f.integral(),
        }
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        match self {
            Filter::Box(f) => f.sample(u),
            Filter::Triangle(f) => f.sample(u),
            Filter::Gaussian(f) => f.sample(u),
            Filter::Mitchell(f) => f.sample(u),
            Filter::LanczosSinc(f) => f.sample(u),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FilterSampler {
    domain: Bounds2f,
    f: Vec2D<Float>,
    distribution: PiecewiseConstant2D,
}

impl FilterSampler {
    pub fn new<F: AbstractFilter>(filter: F) -> FilterSampler {
        let domain = Bounds2f::new(Point2f::from(-filter.radius()), Point2f::from(filter.radius()));
        let mut f = Vec2D::from_bounds(Bounds2i::new(Point2i::ZERO, Point2i::new((32.0 * filter.radius().x) as i32, (32.0 * filter.radius().y) as i32)));

        for y in 0..f.height() {
            for x in 0..f.width() {
                let p = domain.lerp(Point2f::new(
                    (x as Float + 0.5) / f.width() as Float,
                    (y as Float + 0.5) / f.height() as Float,
                ));
                f.set(Point2i::new(x, y), filter.evaluate(p));
            }
        }

        let distribution = PiecewiseConstant2D::new_from_2d(&f, domain);

        FilterSampler {
            domain,
            f,
            distribution,
        }
    }

    pub fn sample(&self, u: Point2f) -> FilterSample {
        let (p, pdf, pi) = self.distribution.sample(u);
        FilterSample { p, weight: self.f.get(pi) / pdf }
    }
}

pub fn get_camera_sample<T: AbstractSampler>(
    sampler: &mut T,
    p_pixel: Point2i,
    filter: &Filter,
    options: &Options,
) -> CameraSample {
    let filter_sample = filter.sample(sampler.get_pixel_2d());
    if options.disable_pixel_jitter {
        CameraSample {
            p_film: Point2f::new(p_pixel.x as f32, p_pixel.y as f32) + Point2f::new(0.5, 0.5),
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            filter_weight: 1.0,
        }
    } else {
        CameraSample {
            p_film: Point2f::new(p_pixel.x as f32, p_pixel.y as f32) + Point2f::from(filter_sample.p) + Point2f::new(0.5, 0.5),
            p_lens: sampler.get_2d(),
            time: sampler.get_1d(),
            filter_weight: filter_sample.weight,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoxFilter {
    radius: Vec2f,
}

impl BoxFilter {
    pub fn create(parameters: &mut ParameterDictionary, loc: &FileLoc) -> BoxFilter {
        let xw = parameters.get_one_float("xradius", 0.5);
        let yw = parameters.get_one_float("yradius", 0.5);

        BoxFilter {
            radius: Vec2f::new(xw, yw),
        }
    }

    pub fn new(radius: Vec2f) -> BoxFilter {
        BoxFilter { radius }
    }
}

impl AbstractFilter for BoxFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        if Float::abs(p.x) <= self.radius.x && Float::abs(p.y) <= self.radius.y {
            1.0
        } else {
            0.0
        }
    }

    fn integral(&self) -> Float {
        2.0 * self.radius.x * 2.0 * self.radius.y
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        let p = Point2f::new(
            lerp(-self.radius.x, self.radius.x, u[0]),
            lerp(-self.radius.y, self.radius.y, u[1]),
        );
        FilterSample { p, weight: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct TriangleFilter {
    radius: Vec2f,
}

impl TriangleFilter {
    pub fn create(parameters: &mut ParameterDictionary, loc: &FileLoc) -> TriangleFilter {
        let xw = parameters.get_one_float("xradius", 2.0);
        let yw = parameters.get_one_float("yradius", 2.0);
        TriangleFilter::new(Vec2f::new(xw, yw))
    }

    pub fn new(radius: Vec2f) -> TriangleFilter {
        TriangleFilter { radius }
    }
}

impl AbstractFilter for TriangleFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        Float::max(0.0, self.radius.x - p.x.abs()) * Float::max(0.0, self.radius.y - p.y.abs())
    }

    fn integral(&self) -> Float {
        sqr(self.radius.x) * sqr(self.radius.y)
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        FilterSample {
            p: Point2f::new(sample_tent(u.x, self.radius.x), sample_tent(u.y, self.radius.y)),
            weight: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GaussianFilter {
    radius: Vec2f,
    sigma: Float,
    exp_x: Float,
    exp_y: Float,
    sampler: FilterSampler,
}

impl GaussianFilter {
    pub fn create(parameters: &mut ParameterDictionary, loc: &FileLoc) -> GaussianFilter {
        let xw = parameters.get_one_float("xradius", 1.5);
        let yw = parameters.get_one_float("yradius", 1.5);
        let sigma = parameters.get_one_float("sigma", 0.5);
        GaussianFilter::new(Vec2f::new(xw, yw), sigma)
    }

    pub fn new(radius: Vec2f, sigma: Float) -> GaussianFilter {
        let mut filter = GaussianFilter {
            radius,
            sigma,
            exp_x: gaussian(radius.x, 0.0, sigma),
            exp_y: gaussian(radius.y, 0.0, sigma),
            sampler: FilterSampler::default()
        };

        filter.sampler = FilterSampler::new(filter.clone());
        filter
    }
}

impl AbstractFilter for GaussianFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        Float::max(0.0, gaussian(p.x, 0.0, self.sigma) - self.exp_x)
            * Float::max(0.0, gaussian(p.y, 0.0, self.sigma) - self.exp_y)
    }

    fn integral(&self) -> Float {
        (gaussian_integral(-self.radius.x, self.radius.x, 0.0, self.sigma) - 2.0 * self.radius.x * self.exp_x)
            * (gaussian_integral(-self.radius.y, self.radius.y, 0.0, self.sigma) - 2.0 * self.radius.y * self.exp_y)
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

#[derive(Debug, Clone)]
pub struct MitchellFilter {
    radius: Vec2f,
    b: Float,
    c: Float,
    sampler: FilterSampler,
}

impl MitchellFilter {
    pub fn create(parameters: &mut ParameterDictionary, loc: &FileLoc) -> MitchellFilter {
        let xw = parameters.get_one_float("xradius", 2.0);
        let yw = parameters.get_one_float("yradius", 2.0);
        let b = parameters.get_one_float("B", 1.0 / 3.0);
        let c = parameters.get_one_float("C", 1.0 / 3.0);
        MitchellFilter::new(Vec2f::new(xw, yw), b, c)
    }

    pub fn new(radius: Vec2f, b: Float, c: Float) -> MitchellFilter {
        let mut filter = MitchellFilter {
            radius,
            b,
            c,
            sampler: FilterSampler::default(),
        };

        filter.sampler = FilterSampler::new(filter.clone());
        filter
    }

    fn mitchell_1d(&self, mut x: Float) -> Float {
        x = x.abs();
        if x <= 1.0 {
            ((12.0 - 9.0 * self.b - 6.0 * self.c) * x * x * x + (-18.0 + 12.0 * self.b + 6.0 * self.c) * x * x
                + (6.0 - 2.0 * self.b))
                * (1.0 / 6.0)
        } else if x <= 2.0 {
            ((-self.b - 6.0 * self.c) * x * x * x + (6.0 * self.b + 30.0 * self.c) * x * x
                + (-12.0 * self.b - 48.0 * self.c) * x + (8.0 * self.b + 24.0 * self.c))
                * (1.0 / 6.0)
        } else {
            0.0
        }
    }
}

impl AbstractFilter for MitchellFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        self.mitchell_1d(2.0 * p.x / self.radius.x) * self.mitchell_1d(2.0 * p.y / self.radius.y)
    }

    fn integral(&self) -> Float {
        self.radius.x * self.radius.y / 4.0
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

#[derive(Debug, Clone)]
pub struct LanczosSincFilter {
    radius: Vec2f,
    tau: Float,
    sampler: FilterSampler,
}

impl LanczosSincFilter {
    pub fn create(parameters: &mut ParameterDictionary, loc: &FileLoc) -> LanczosSincFilter {
        let xw = parameters.get_one_float("xradius", 4.0);
        let yw = parameters.get_one_float("yradius", 4.0);
        let tau = parameters.get_one_float("tau", 3.0);
        LanczosSincFilter::new(Vec2f::new(xw, yw), tau)
    }

    pub fn new(radius: Vec2f, tau: Float) -> LanczosSincFilter {
        let mut filter = LanczosSincFilter {
            radius,
            tau,
            sampler: FilterSampler::default(),
        };

        filter.sampler = FilterSampler::new(filter.clone());
        filter
    }
}

impl AbstractFilter for LanczosSincFilter {
    fn radius(&self) -> Vec2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        windowed_sinc(p.x, self.radius.x, self.tau) * windowed_sinc(p.y, self.radius.y, self.tau)
    }

    fn integral(&self) -> Float {
        const SQRT_SAMPLES: usize = 64;
        const N_SAMPLES: usize = SQRT_SAMPLES * SQRT_SAMPLES;

        let mut sum = 0.0;
        let area = 2.0 * self.radius.x * 2.0 * self.radius.y;
        let mut rng = rand::thread_rng();

        for y in 0..SQRT_SAMPLES {
            for x in 0..SQRT_SAMPLES {
                let u = Point2f::new((x as Float + rng.gen::<Float>()) / SQRT_SAMPLES as Float, (y as Float + rng.gen::<Float>()) / SQRT_SAMPLES as Float);
                let p = Point2f::new(lerp(-self.radius.x, self.radius.x, u.x), lerp(-self.radius.y, self.radius.y, u.y));
                sum += self.evaluate(p);
            }
        }

        sum / N_SAMPLES as Float * area
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        self.sampler.sample(u)
    }
}

pub struct FilterSample {
    pub p: Point2f,
    pub weight: Float,
}
