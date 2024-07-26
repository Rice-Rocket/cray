use crate::{lerp, Point2f, Float, Vec2f};

pub trait AbstractFilter {
    fn radius(&self) -> Vec2f;

    fn evaluate(&self, p: Point2f) -> Float;

    fn integral(&self) -> Float;

    fn sample(&self, u: Point2f) -> FilterSample;
}

#[derive(Debug, Clone)]
pub enum Filter {
    BoxFilter(BoxFilter)
}

impl Filter {
    pub fn create(name: &str, /* parameters: &mut ParameterDictionary, loc: &FileLoc */) -> Filter {
        match name {
            "box" => Filter::BoxFilter(BoxFilter::create(/* parameters, loc */)),
            _ => panic!("unknown filter type")
        }
    }
}

impl AbstractFilter for Filter {
    fn radius(&self) -> Vec2f {
        match self {
            Filter::BoxFilter(f) => f.radius(),
        }
    }

    fn evaluate(&self, p: Point2f) -> Float {
        match self {
            Filter::BoxFilter(f) => f.evaluate(p),
        }
    }

    fn integral(&self) -> Float {
        match self {
            Filter::BoxFilter(f) => f.integral(),
        }
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        match self {
            Filter::BoxFilter(f) => f.sample(u),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoxFilter {
    radius: Vec2f,
}

impl BoxFilter {
    pub fn create(/* parameters: &mut ParameterDictionary, loc: &FileLoc */) -> BoxFilter {
        // let xw = parameters.get_one_float("xradius", 0.5);
        // let yw = parameters.get_one_float("yradius", 0.5);
        let xw = 0.5;
        let yw = 0.5;

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

pub struct FilterSample {
    pub p: Point2f,
    pub weight: Float,
}
