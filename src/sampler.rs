use hexf::hexf32;
use num::integer::Roots;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::{error, hash, hashing::{combine_bits_64, encode_morton_2, ihash21, ihash41, mix_bits, mix_bits_32, mur, permutation_element, split_u64}, lowdiscrepancy::{compute_radical_inverse_permutations, inverse_radical_inverse, owen_scrambled_radical_inverse, radical_inverse, scrambled_radical_inverse, sobol_sample, BinaryPermuteScrambler, DigitPermutation, FastOwenScrambler, NoScrambler, OwenScrambler}, modulo, options::Options, primes::PRIME_TABLE_SIZE, reader::{error::ParseResult, paramdict::ParameterDictionary, target::FileLoc}, round_up_pow_2, transmute, warn, Float, Point2f, Point2i, Vec2u, ONE_MINUS_EPSILON};

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
    Halton(HaltonSampler),
    ZSobol(ZSobolSampler),
}

impl Sampler {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        full_res: Point2i,
        options: &Options,
        loc: &FileLoc,
    ) -> ParseResult<Sampler> {
        Ok(match name {
            "zsobol" => Sampler::ZSobol(ZSobolSampler::create(parameters, full_res, options, loc)?),
            // "paddedsobol" => Sampler::PaddedSobol(PaddedSobolSampler::create(parameters, options, loc)),
            "halton" => Sampler::Halton(HaltonSampler::create(parameters, full_res, options, loc)?),
            // "sobol" => Sampler::Sobol(SobolSampler::create(parameters, options, loc)),
            // "pmj02bn" => Sampler::PMJ02BN(PMJ02BNSampler::create(parameters, options, loc)),
            "independent" => Sampler::Independent(IndependentSampler::create(parameters, options, loc)?),
            "stratified" => Sampler::Stratified(StratifiedSampler::create(parameters, options, loc)?),
            _ => { error!(loc, UnknownValue, "unknown sampler type '{}'", name); }
        })
    }
}

impl AbstractSampler for Sampler {
    fn samples_per_pixel(&self) -> i32 {
        match self {
            Sampler::Independent(s) => s.samples_per_pixel(),
            Sampler::Stratified(s) => s.samples_per_pixel(),
            Sampler::Halton(s) => s.samples_per_pixel(),
            Sampler::ZSobol(s) => s.samples_per_pixel(),
        }
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32) {
       match self {
           Sampler::Independent(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::Stratified(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::Halton(s) => s.start_pixel_sample(p, sample_index, dimension),
           Sampler::ZSobol(s) => s.start_pixel_sample(p, sample_index, dimension),
       }
    }

    fn get_1d(&mut self) -> Float {
        match self {
            Sampler::Independent(s) => s.get_1d(),
            Sampler::Stratified(s) => s.get_1d(),
            Sampler::Halton(s) => s.get_1d(),
            Sampler::ZSobol(s) => s.get_1d(),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_2d(),
            Sampler::Stratified(s) => s.get_2d(),
            Sampler::Halton(s) => s.get_2d(),
            Sampler::ZSobol(s) => s.get_2d(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_pixel_2d(),
            Sampler::Stratified(s) => s.get_pixel_2d(),
            Sampler::Halton(s) => s.get_pixel_2d(),
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
    ) -> ParseResult<IndependentSampler> {
        let mut ns = parameters.get_one_int("pixelsamples", 4)?;
        if let Some(pixel_samples) = options.pixel_samples {
            ns = pixel_samples;
        }

        let seed = parameters.get_one_int("seed", options.seed)? as u64;
        Ok(IndependentSampler::new(seed, ns))
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
    ) -> ParseResult<StratifiedSampler> {
        let jitter = parameters.get_one_bool("jitter", true)?;
        let mut x_samples = parameters.get_one_int("xsamples", 4)?;
        let mut y_samples = parameters.get_one_int("ysamples", 4)?;
        
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

        let seed = parameters.get_one_int("seed", options.seed)? as u32;
        Ok(StratifiedSampler::new(x_samples, y_samples, jitter, seed))
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
        let delta = if self.jitter { self.rng.gen() } else { 0.5 };
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
pub struct HaltonSampler {
    samples_per_pixel: i32,
    randomize: RandomizeStrategy,
    digit_permutations: Vec<DigitPermutation>,
    base_scales: Point2i,
    base_exponents: Point2i,
    mul_inverse: [i32; 2],
    halton_index: u64,
    dimension: u32,
}

impl HaltonSampler {
    pub const MAX_HALTON_RESOLUTION: i32 = 128;

    pub fn create(
        parameters: &mut ParameterDictionary,
        full_res: Point2i,
        options: &Options,
        loc: &FileLoc,
    ) -> ParseResult<HaltonSampler> {
        let mut n_samples = parameters.get_one_int("pixelsamples", 16)?;
        if let Some(s) = options.pixel_samples {
            n_samples = s;
        }

        let seed = parameters.get_one_int("seed", options.seed)?;
        if options.quick_render {
            n_samples = 1;
        }

        let random = parameters.get_one_string("randomization", "permutedigits")?;
        let randomizer = match random.as_ref() {
            "none" => RandomizeStrategy::None,
            "permutedigits" => RandomizeStrategy::PermuteDigits,
            "fastowen" => { error!(loc, InvalidValue, "'fastowen' randomization not supported by HaltonSampler."); },
            "owen" => RandomizeStrategy::Owen,
            _ => { error!(loc, UnknownValue, "unknown randomization '{}'", random); }
        };

        Ok(HaltonSampler::new(
            n_samples,
            full_res,
            randomizer,
            seed,
        ))
    }

    pub fn new(
        samples_per_pixel: i32,
        full_res: Point2i,
        randomize: RandomizeStrategy,
        seed: i32,
    ) -> HaltonSampler {
        let digit_permutations = if let RandomizeStrategy::PermuteDigits = randomize {
            compute_radical_inverse_permutations(seed)
        } else {
            Vec::new()
        };

        let mut base_scales = Point2i::ZERO;
        let mut base_exponents = Point2i::ZERO;

        for i in 0..2 {
            let base = if i == 0 { 2 } else { 3 };
            let mut scale = 1;
            let mut exp = 0;

            while scale < full_res[i].min(Self::MAX_HALTON_RESOLUTION) {
                scale *= base;
                exp += 1;
            }

            base_scales[i] = scale;
            base_exponents[i] = exp;
        }

        let mul_inverse = [
            Self::multiplicative_inverse(base_scales[1] as i64, base_scales[0] as i64) as i32,
            Self::multiplicative_inverse(base_scales[0] as i64, base_scales[1] as i64) as i32,
        ];

        HaltonSampler {
            samples_per_pixel,
            randomize,
            digit_permutations,
            base_scales,
            base_exponents,
            mul_inverse,
            halton_index: 0,
            dimension: 0,
        }
    }

    fn sample_dimension(&self, dimension: u32) -> Float {
        match self.randomize {
            RandomizeStrategy::None => radical_inverse(dimension, self.halton_index),
            RandomizeStrategy::PermuteDigits => scrambled_radical_inverse(dimension, self.halton_index, &self.digit_permutations[dimension as usize]),
            RandomizeStrategy::Owen => owen_scrambled_radical_inverse(dimension, self.halton_index, mix_bits(1 + ((dimension as u64) << 4)) as u32),
            RandomizeStrategy::FastOwen => unreachable!(),
        }
    }

    fn multiplicative_inverse(a: i64, n: i64) -> u64 {
        let (x, y) = Self::extended_gcd(a, n);
        x.rem_euclid(n) as u64
    }

    fn extended_gcd(a: i64, b: i64) -> (i64, i64) {
        if b == 0 {
            return (1, 0);
        }

        let d = a / b;
        let (xp, yp) = Self::extended_gcd(b, a % b);
        (yp, xp - (d * yp))
    }
}

impl AbstractSampler for HaltonSampler {
    fn samples_per_pixel(&self) -> i32 {
        self.samples_per_pixel
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: u32) {
        self.halton_index = 0;
        let sample_stride = self.base_scales[0] * self.base_scales[1];

        if sample_stride > 1 {
            let pm = Point2i::new(p[0].rem_euclid(Self::MAX_HALTON_RESOLUTION), p[1].rem_euclid(Self::MAX_HALTON_RESOLUTION));

            for i in 0..2 {
                let dim_offset = if i == 0 {
                    inverse_radical_inverse(pm[i] as u64, 2, self.base_exponents[i])
                } else {
                    inverse_radical_inverse(pm[i] as u64, 3, self.base_exponents[i])
                };

                self.halton_index += dim_offset * ((sample_stride / self.base_scales[i]) * self.mul_inverse[i]) as u64;
            }

            self.halton_index %= sample_stride as u64;
        }

        self.halton_index += (sample_index * sample_stride) as u64;
        self.dimension = u32::max(2, dimension);
    }

    fn get_1d(&mut self) -> Float {
        if self.dimension >= PRIME_TABLE_SIZE {
            self.dimension = 2;
        }

        let dim = self.dimension;
        self.dimension += 1;

        self.sample_dimension(dim)
    }

    fn get_2d(&mut self) -> Point2f {
        if self.dimension + 1 >= PRIME_TABLE_SIZE {
            self.dimension = 2;
        }

        let dim = self.dimension;
        self.dimension += 2;

        Point2f::new(self.sample_dimension(dim), self.sample_dimension(dim + 1))
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        Point2f::new(
            radical_inverse(0, self.halton_index >> self.base_exponents[0]),
            radical_inverse(1, self.halton_index / self.base_scales[1] as u64),
        )
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
    ) -> ParseResult<ZSobolSampler> {
        let mut n_samples = parameters.get_one_int("pixelsamples", 16)?;
        if let Some(samples) = options.pixel_samples {
            n_samples = samples;
        }

        if options.quick_render {
            n_samples = 1;
        }

        let seed = parameters.get_one_int("seed", options.seed)?;

        let randomize = match parameters.get_one_string("randomization", "fastowen")?.as_str() {
            "none" => RandomizeStrategy::None,
            "permutedigits" => RandomizeStrategy::PermuteDigits,
            "fastowen" => RandomizeStrategy::FastOwen,
            "owen" => RandomizeStrategy::Owen,
            s => { error!(loc, UnknownValue, "unknown randomization strategy given to ZSobolSampler '{}'", s); },
        };

        Ok(ZSobolSampler::new(n_samples, full_res, randomize, seed))
    }

    pub fn new(
        samples_per_pixel: i32,
        full_res: Point2i,
        randomize: RandomizeStrategy,
        seed: i32,
    ) -> ZSobolSampler {
        if (samples_per_pixel & (samples_per_pixel - 1)) != 0 {
            warn!(@basic "sobol samplers with non power-of-two samples count ({}) are suboptimal", samples_per_pixel);
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
            RandomizeStrategy::None => sobol_sample::<_, 0>(sample_index, NoScrambler),
            RandomizeStrategy::PermuteDigits => sobol_sample::<_, 0>(sample_index, BinaryPermuteScrambler { permutation: sample_hash }),
            RandomizeStrategy::FastOwen => sobol_sample::<_, 0>(sample_index, FastOwenScrambler { seed: sample_hash }),
            RandomizeStrategy::Owen => sobol_sample::<_, 0>(sample_index, OwenScrambler { seed: sample_hash }),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        let sample_index = self.get_sample_index();
        self.dimension += 2;

        let bits = mix_bits(combine_bits_64(self.dimension as u64, self.seed as u64));
        let sample_hash = split_u64(bits);

        match self.randomize {
            RandomizeStrategy::None => Point2f::new(
                sobol_sample::<_, 0>(sample_index, NoScrambler),
                sobol_sample::<_, 1>(sample_index, NoScrambler),
            ),
            RandomizeStrategy::PermuteDigits => Point2f::new(
                sobol_sample::<_, 0>(sample_index, BinaryPermuteScrambler { permutation: sample_hash.0 }),
                sobol_sample::<_, 1>(sample_index, BinaryPermuteScrambler { permutation: sample_hash.1 }),
            ),
            RandomizeStrategy::FastOwen => Point2f::new(
                sobol_sample::<_, 0>(sample_index, FastOwenScrambler { seed: sample_hash.0 }),
                sobol_sample::<_, 1>(sample_index, FastOwenScrambler { seed: sample_hash.1 }),
            ),
            RandomizeStrategy::Owen => Point2f::new(
                sobol_sample::<_, 0>(sample_index, OwenScrambler { seed: sample_hash.0 }),
                sobol_sample::<_, 1>(sample_index, OwenScrambler { seed: sample_hash.1 }),
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
