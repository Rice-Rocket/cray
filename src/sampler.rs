use num::integer::Roots;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{options::Options, reader::{paramdict::ParameterDictionary, target::FileLoc}, Float, Point2f, Point2i, Vec2u};

pub trait AbstractSampler {
    fn samples_per_pixel(&self) -> i32;

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32);

    fn get_1d(&mut self) -> Float;
    fn get_2d(&mut self) -> Point2f;
    fn get_pixel_2d(&mut self) -> Point2f;
}

#[derive(Debug, Clone)]
pub enum Sampler {
    Independent(IndependentSampler),
    Stratified(StratifiedSampler),
}

impl Sampler {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        full_res: Point2i,
        options: &Options,
        loc: &FileLoc,
    ) -> Sampler {
        match name {
            // "zsobol" => Sampler::ZSobol(ZSobolSampler::create(parameters, options, loc)),
            // "paddedsobol" => Sampler::PaddedSobol(PaddedSobolSampler::create(parameters, options, loc)),
            // "halton" => Sampler::Halton(HaltonSampler::create(parameters, options, loc)),
            // "sobol" => Sampler::Sobol(SobolSampler::create(parameters, options, loc)),
            // "pmj02bn" => Sampler::PMJ02BN(PMJ02BNSampler::create(parameters, options, loc)),
            "independent" => Sampler::Independent(IndependentSampler::create(parameters, options, loc)),
            "stratified" => Sampler::Stratified(StratifiedSampler::create(parameters, options, loc)),
            _ => panic!("unknown sampler type {}", name)
        }
    }
}

impl AbstractSampler for Sampler {
    fn samples_per_pixel(&self) -> i32 {
        match self {
            Sampler::Independent(s) => s.samples_per_pixel(),
            Sampler::Stratified(s) => s.samples_per_pixel(),
        }
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32) {
       match self {
           Sampler::Independent(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::Stratified(s) => s.start_pixel_sample(p, sample_index, dimension),
       }
    }

    fn get_1d(&mut self) -> Float {
        match self {
            Sampler::Independent(s) => s.get_1d(),
            Sampler::Stratified(s) => s.get_1d(),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_2d(),
            Sampler::Stratified(s) => s.get_2d(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_pixel_2d(),
            Sampler::Stratified(s) => s.get_pixel_2d(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndependentSampler {
    seed: u64,
    samples_per_pixel: i32,
    rng: SmallRng,
}

impl IndependentSampler {
    pub fn new(seed: u64, samples_per_pixel: i32) -> IndependentSampler {
        IndependentSampler {
            seed,
            samples_per_pixel,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn create(
        parameters: &mut ParameterDictionary,
        options: &Options,
        loc: &FileLoc,
    ) -> IndependentSampler {
        let mut ns = parameters.get_one_int("pixelsamples", 4);
        if let Some(pixel_samples) = options.pixel_samples {
            ns = pixel_samples;
        }

        let seed = parameters.get_one_int("seed", options.seed) as u64;
        IndependentSampler::new(seed, ns)
    }
}

impl AbstractSampler for IndependentSampler {
    fn samples_per_pixel(&self) -> i32 {
        self.samples_per_pixel
    }

    fn start_pixel_sample(&mut self, _p: Point2i, _sample_index: i32, _dimension: i32) {
        // TODO: Make the RNG deterministic
    }

    fn get_1d(&mut self) -> Float {
        self.rng.gen()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f {
            x: self.rng.gen(),
            y: self.rng.gen(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }
}

#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    x_pixel_samples: i32,
    y_pixel_samples: i32,
    seed: u64,
    jitter: bool,
    rng: SmallRng,
    pixel: Point2i,
    sample_index: i32,
    dimension: i32,
}

impl StratifiedSampler {
    pub fn new(x_samples: i32, y_samples: i32, jitter: bool, seed: u64) -> StratifiedSampler {
        StratifiedSampler {
            x_pixel_samples: x_samples,
            y_pixel_samples: y_samples,
            seed,
            jitter,
            rng: SmallRng::seed_from_u64(seed),
            pixel: Point2i::default(),
            sample_index: 0,
            dimension: 0,
        }
    }

    pub fn create(
        parameters: &mut ParameterDictionary,
        options: &Options,
        loc: &FileLoc,
    ) -> StratifiedSampler {
        let jitter = parameters.get_one_bool("jitter", true);
        let mut x_samples = parameters.get_one_int("xsamples", 4);
        let mut y_samples = parameters.get_one_int("ysamples", 4);
        
        if let Some(n_samples) = options.pixel_samples {
            let mut div = n_samples.sqrt();
            while n_samples % div != 0 {
                debug_assert!(div > 0);
                div -= 1;
            }

            x_samples = n_samples / div;
            y_samples = n_samples / x_samples;
            debug_assert_eq!(n_samples, x_samples * y_samples);
        }

        if options.quick_render {
            x_samples = 1;
            y_samples = 1;
        }

        let seed = parameters.get_one_int("seed", options.seed) as u64;
        StratifiedSampler::new(x_samples, y_samples, jitter, seed)
    }
}

impl AbstractSampler for StratifiedSampler {
    fn samples_per_pixel(&self) -> i32 {
        self.x_pixel_samples * self.y_pixel_samples
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32) {
        self.pixel = p;
        self.sample_index = sample_index;
        self.dimension = dimension;
        // TODO: Make the RNG deterministic
    }

    fn get_1d(&mut self) -> Float {
        let hash = ihash21(ihash21(self.pixel.x as u32, self.pixel.y as u32), ihash21(self.dimension as u32, self.seed as u32));
        let stratum = permutation_element(self.sample_index as u32, self.samples_per_pixel() as u32, hash);

        self.dimension += 1;
        let delta= if self.jitter { self.rng.gen() } else { 0.5 };
        (stratum as Float + delta) / self.samples_per_pixel() as Float
    }

    fn get_2d(&mut self) -> Point2f {
        let hash = ihash21(ihash21(self.pixel.x as u32, self.pixel.y as u32), ihash21(self.dimension as u32, self.seed as u32));
        let stratum = permutation_element(self.sample_index as u32, self.samples_per_pixel() as u32, hash);

        self.dimension += 2;
        let x = stratum % self.x_pixel_samples;
        let y = stratum / self.x_pixel_samples;
        let dx = if self.jitter { self.rng.gen() } else { 0.5 };
        let dy = if self.jitter { self.rng.gen() } else { 0.5 };
        Point2f::new(
            (x as Float + dx) / self.x_pixel_samples as Float,
            (y as Float + dy) / self.y_pixel_samples as Float,
        )
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }
}

fn ihash21(mut x: u32, mut y: u32) -> u32 {
    x = x.wrapping_mul(73333);
    y = y.wrapping_mul(7777);
    x ^= 3333777777u32.wrapping_shr(x.wrapping_shr(28));
    y ^= 3333777777u32.wrapping_shr(y.wrapping_shr(28));
    let n = x.wrapping_mul(y);
    n ^ (n.wrapping_shr(15))
}

fn permutation_element(mut i: u32, l: u32, p: u32) -> i32 {
    let mut w = l - 1;
    w |= w.wrapping_shr(1);
    w |= w.wrapping_shr(2);
    w |= w.wrapping_shr(4);
    w |= w.wrapping_shr(8);
    w |= w.wrapping_shr(16);

    while {
        i ^= p;
        i = i.wrapping_mul(0xe170893d);
        i ^= p.wrapping_shr(16);
        i ^= (i & w).wrapping_shr(4);
        i ^= p.wrapping_shr(8);
        i = i.wrapping_mul(0x0929eb3f);
        i ^= p.wrapping_shr(23);
        i ^= (i & w).wrapping_shr(1);
        i = i.wrapping_mul(1 | p.wrapping_shr(27));
        i = i.wrapping_mul(0x6935fa69);
        i ^= (i & w).wrapping_shr(11);
        i = i.wrapping_mul(0x74dcb303);
        i ^= (i & w).wrapping_shr(2);
        i = i.wrapping_mul(0x9e501cc3);
        i ^= (i & w).wrapping_shr(2);
        i = i.wrapping_mul(0xc860a3df);
        i &= w;
        i ^= i.wrapping_shr(5);

        i >= l
    } {}

    ((i.wrapping_add(p)) % l) as i32
}
