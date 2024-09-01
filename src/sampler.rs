use hexf::hexf32;
use num::integer::Roots;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tracing::warn;

use crate::{hash, hashing::{combine_bits_64, encode_morton_2, fmix, ihash21, ihash41, mix_bits, mur, permutation_element, split_u64}, lowdiscrepancy::{sobol_sample, BinaryPermuteScrambler, FastOwenScrambler, NoScrambler, OwenScrambler}, options::Options, reader::{paramdict::ParameterDictionary, target::FileLoc}, round_up_pow_2, transmute, Float, Point2f, Point2i, Vec2u, ONE_MINUS_EPSILON};

pub trait AbstractSampler {
    fn samples_per_pixel(&self) -> i32;

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32);

    fn get_1d(&mut self) -> Float;
    fn get_2d(&mut self) -> Point2f;
    fn get_pixel_2d(&mut self) -> Point2f;
}

#[derive(Debug, Clone)]
pub enum Sampler {
    Independent(IndependentSampler),
    Stratified(StratifiedSampler),
    ZSobol(ZSobolSampler),
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
            "zsobol" => Sampler::ZSobol(ZSobolSampler::create(parameters, full_res, options, loc)),
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
            Sampler::ZSobol(s) => s.samples_per_pixel(),
        }
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32) {
       match self {
           Sampler::Independent(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::Stratified(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::ZSobol(s) => s.start_pixel_sample(p, sample_index, dimension),
       }
    }

    fn get_1d(&mut self) -> Float {
        match self {
            Sampler::Independent(s) => s.get_1d(),
            Sampler::Stratified(s) => s.get_1d(),
            Sampler::ZSobol(s) => s.get_1d(),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_2d(),
            Sampler::Stratified(s) => s.get_2d(),
            Sampler::ZSobol(s) => s.get_2d(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_pixel_2d(),
            Sampler::Stratified(s) => s.get_pixel_2d(),
            Sampler::ZSobol(s) => s.get_pixel_2d(),
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

    fn start_pixel_sample(&mut self, _p: Point2i, _sample_index: i32, _dimension: u32) {
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
    seed: u32,
    jitter: bool,
    rng: SmallRng,
    pixel: Point2i,
    sample_index: i32,
    dimension: u32,
}

impl StratifiedSampler {
    pub fn new(x_samples: i32, y_samples: i32, jitter: bool, seed: u32) -> StratifiedSampler {
        StratifiedSampler {
            x_pixel_samples: x_samples,
            y_pixel_samples: y_samples,
            seed,
            jitter,
            rng: SmallRng::seed_from_u64(seed as u64),
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

        let seed = parameters.get_one_int("seed", options.seed) as u32;
        StratifiedSampler::new(x_samples, y_samples, jitter, seed)
    }
}

impl AbstractSampler for StratifiedSampler {
    fn samples_per_pixel(&self) -> i32 {
        self.x_pixel_samples * self.y_pixel_samples
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32) {
        self.pixel = p;
        self.sample_index = sample_index;
        self.dimension = dimension;
        // TODO: Make the RNG deterministic
    }

    fn get_1d(&mut self) -> Float {
        let seed = self.seed ^ self.dimension;
        let hash = ihash21(
            transmute!(self.pixel.x => u32) ^ seed,
            transmute!(self.pixel.y => u32) ^ seed,
        );
        let stratum = permutation_element(self.sample_index as u32, self.samples_per_pixel() as u32, hash as u32);

        self.dimension += 1;
        let delta= if self.jitter { self.rng.gen() } else { 0.5 };
        (stratum as Float + delta) / self.samples_per_pixel() as Float
    }

    fn get_2d(&mut self) -> Point2f {
        let seed = self.seed ^ self.dimension;
        let hash = ihash21(
            transmute!(self.pixel.x => u32) ^ seed,
            transmute!(self.pixel.y => u32) ^ seed,
        );
        let stratum = permutation_element(self.sample_index as u32, self.samples_per_pixel() as u32, hash as u32);

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

#[derive(Debug, Clone)]
pub struct ZSobolSampler {
    randomize: RandomizeStrategy,
    seed: u32,
    log2_samples_per_pixel: i32,
    n_base_4_digits: i32,
    morton_index: u64,
    dimension: u32,
}

impl ZSobolSampler {
    pub fn create(
        parameters: &mut ParameterDictionary,
        full_res: Point2i,
        options: &Options,
        loc: &FileLoc,
    ) -> ZSobolSampler {
        let mut n_samples = parameters.get_one_int("pixelsamples", 16);
        if let Some(samples) = options.pixel_samples {
            n_samples = samples;
        }

        if options.quick_render {
            n_samples = 1;
        }

        let seed = parameters.get_one_int("seed", options.seed);

        let randomize = match parameters.get_one_string("randomization", "fastowen").as_str() {
            "none" => RandomizeStrategy::None,
            "permutedigits" => RandomizeStrategy::PermuteDigits,
            "fastowen" => RandomizeStrategy::FastOwen,
            "owen" => RandomizeStrategy::Owen,
            s => panic!("unknown randomization strategy given to ZSobolSampler {}", s),
        };

        ZSobolSampler::new(n_samples, full_res, randomize, seed)
    }

    pub fn new(
        samples_per_pixel: i32,
        full_res: Point2i,
        randomize: RandomizeStrategy,
        seed: i32,
    ) -> ZSobolSampler {
        if (samples_per_pixel & (samples_per_pixel - 1)) != 0 {
            warn!("sobol samplers with non power-of-two samples count ({}) are suboptimal", samples_per_pixel);
        }

        let log2_samples_per_pixel = samples_per_pixel.ilog2();
        let res = round_up_pow_2(full_res.x.max(full_res.y));
        let log4_samples_per_pixel = (log2_samples_per_pixel + 1) / 2;
        let n_base_4_digits = res.ilog2() + log4_samples_per_pixel;

        ZSobolSampler {
            randomize,
            seed: seed as u32,
            log2_samples_per_pixel: log2_samples_per_pixel as i32,
            n_base_4_digits: n_base_4_digits as i32,
            morton_index: 0,
            dimension: 0,
        }
    }

    fn get_sample_index(&self) -> u64 {
        const PERMUTATIONS: [u8; 96] = [
            0, 1, 2, 3,
            0, 1, 3, 2,
            0, 2, 1, 3,
            0, 2, 3, 1,
            0, 3, 2, 1,
            0, 3, 1, 2,
            1, 0, 2, 3,
            1, 0, 3, 2,
            1, 2, 0, 3,
            1, 2, 3, 0,
            1, 3, 2, 0,
            1, 3, 0, 2,
            2, 1, 0, 3,
            2, 1, 3, 0,
            2, 0, 1, 3,
            2, 0, 3, 1,
            2, 3, 0, 1,
            2, 3, 1, 0,
            3, 1, 2, 0,
            3, 1, 0, 2,
            3, 2, 1, 0,
            3, 2, 0, 1,
            3, 0, 2, 1,
            3, 0, 1, 2,
        ];

        let mut sample_index = 0u64;
        let last_digit = self.log2_samples_per_pixel & 1;
        let pow_2_samples = last_digit != 0;

        for i in (last_digit..=self.n_base_4_digits - 1).rev() {
            let digit_shift = 2 * i - last_digit;
            let mut digit = (self.morton_index >> digit_shift) & 3;
            let higher_digits = self.morton_index >> (digit_shift + 2);
            let p = (mix_bits(higher_digits ^ (0x55555555 + self.dimension as u64)) >> 24) % 24;

            digit = *unsafe { PERMUTATIONS.get_unchecked(((p << 2) + digit) as usize) } as u64;
            sample_index |= digit << digit_shift;
        }

        if pow_2_samples {
            let digit = self.morton_index & 1;
            sample_index |= digit ^ (mix_bits((self.morton_index >> 1) ^ (0x55555555 * self.dimension as u64)) & 1);
        }

        sample_index
    }
}

impl AbstractSampler for ZSobolSampler {
    fn samples_per_pixel(&self) -> i32 {
        1 << self.log2_samples_per_pixel
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32) {
        self.dimension = dimension;
        self.morton_index = (encode_morton_2(p.x as u32, p.y as u32) << self.log2_samples_per_pixel) | sample_index as u64;
    }

    fn get_1d(&mut self) -> Float {
        let sample_index = self.get_sample_index();
        self.dimension += 1;

        let sample_hash = ihash21(self.dimension, self.seed);

        match self.randomize {
            RandomizeStrategy::None => sobol_sample(sample_index, 0, NoScrambler),
            RandomizeStrategy::PermuteDigits => sobol_sample(sample_index, 0, BinaryPermuteScrambler { permutation: sample_hash }),
            RandomizeStrategy::FastOwen => sobol_sample(sample_index, 0, FastOwenScrambler { seed: sample_hash }),
            RandomizeStrategy::Owen => sobol_sample(sample_index, 0, OwenScrambler { seed: sample_hash }),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        let sample_index = self.get_sample_index();
        self.dimension += 2;

        let bits = mix_bits(combine_bits_64(self.dimension as u64, self.seed as u64));
        let sample_hash = split_u64(bits);

        match self.randomize {
            RandomizeStrategy::None => Point2f::new(
                sobol_sample(sample_index, 0, NoScrambler),
                sobol_sample(sample_index, 1, NoScrambler),
            ),
            RandomizeStrategy::PermuteDigits => Point2f::new(
                sobol_sample(sample_index, 0, BinaryPermuteScrambler { permutation: sample_hash.0 }),
                sobol_sample(sample_index, 1, BinaryPermuteScrambler { permutation: sample_hash.1 }),
            ),
            RandomizeStrategy::FastOwen => Point2f::new(
                sobol_sample(sample_index, 0, FastOwenScrambler { seed: sample_hash.0 }),
                sobol_sample(sample_index, 1, FastOwenScrambler { seed: sample_hash.1 }),
            ),
            RandomizeStrategy::Owen => Point2f::new(
                sobol_sample(sample_index, 0, OwenScrambler { seed: sample_hash.0 }),
                sobol_sample(sample_index, 1, OwenScrambler { seed: sample_hash.1 }),
            ),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }
}

#[derive(Debug, Clone)]
pub enum RandomizeStrategy {
    None,
    PermuteDigits,
    FastOwen,
    Owen,
}
