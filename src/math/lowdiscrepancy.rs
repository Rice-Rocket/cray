use hexf::hexf32;

use crate::{hashing::{ihash31, mix_bits, permutation_element}, primes::{PRIMES, PRIME_TABLE_SIZE}, sobol::{N_SOBOL_DIMENSIONS, SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE}, transmute, Float, ONE_MINUS_EPSILON};

pub trait AbstractScrambler {
    fn scramble(&self, v: u32) -> u32;
}

pub struct NoScrambler;

impl AbstractScrambler for NoScrambler {
    #[inline]
    fn scramble(&self, v: u32) -> u32 {
        v
    }
}

pub struct BinaryPermuteScrambler {
    pub permutation: u32,
}

impl AbstractScrambler for BinaryPermuteScrambler {
    #[inline]
    fn scramble(&self, v: u32) -> u32 {
        self.permutation ^ v
    }
}

pub struct FastOwenScrambler {
    pub seed: u32
}

impl AbstractScrambler for FastOwenScrambler {
    #[inline]
    fn scramble(&self, mut v: u32) -> u32 {
        v = v.reverse_bits();
        v ^= v.wrapping_mul(0x3d20adea);
        v = v.wrapping_add(self.seed);
        v = v.wrapping_mul(self.seed.wrapping_shr(16) | 1);
        v ^= v.wrapping_mul(0x05526c56);
        v ^= v.wrapping_mul(0x53a22864);
        v.reverse_bits()
    }
}

pub struct OwenScrambler {
    pub seed: u32
}

impl AbstractScrambler for OwenScrambler {
    #[inline]
    fn scramble(&self, mut v: u32) -> u32 {
        if self.seed & 1 != 0 {
            v ^= 1 << 31;
        }

        for b in 1..32 {
            let mask = (!0) << (32 - b);
            if mix_bits(((v & mask) ^ self.seed) as u64) & (1 << b) != 0 {
                v ^= 1 << (31 - b);
            }
        }

        v
    }
}

#[derive(Debug, Clone)]
pub struct DigitPermutation {
    base: u32,
    n_digits: u32,
    permutations: Vec<u16>,
}

impl DigitPermutation {
    pub fn new(base: u32, seed: i32) -> DigitPermutation {
        debug_assert!(base < 65535);

        let mut n_digits = 0;
        let inv_base = 1.0 / base as Float;
        let mut inv_base_m = 1.0;

        while 1.0 - (base - 1) as Float * inv_base_m < 1.0 {
            n_digits += 1;
            inv_base_m *= inv_base;
        }

        let mut permutations = vec![0u16; (n_digits * base) as usize];

        for digit_index in 0..n_digits {
            let dseed = ihash31(
                transmute!(base => u32),
                transmute!(digit_index => u32),
                transmute!(seed => u32),
            );

            for digit_value in 0..base {
                let index = digit_index * base + digit_value;
                permutations[index as usize] = permutation_element(
                    transmute!(digit_value => u32),
                    transmute!(base => u32),
                    dseed,
                ) as u16;
            }
        }

        DigitPermutation {
            base,
            n_digits,
            permutations,
        }
    }

    #[inline]
    pub fn permute(&self, digit_index: u32, digit_value: u32) -> u16 {
        debug_assert!(digit_index < self.n_digits);
        debug_assert!(digit_value < self.base);
        self.permutations[(digit_index * self.base + digit_value) as usize]
    }
}

#[inline]
pub fn sobol_sample<R: AbstractScrambler, const D: usize>(mut a: u64, randomizer: R) -> Float {
    debug_assert!(D < N_SOBOL_DIMENSIONS as usize);
    debug_assert!(a < (1u64 << SOBOL_MATRIX_SIZE));

    let mut v = 0u32;
    let i = unsafe { SOBOL_MATRICES_32.as_ptr().add(D * SOBOL_MATRIX_SIZE as usize) };

    while a != 0 {
        // Index into sobol matrices with the number of trailing zeros 
        // in the binary representation of `a`
        v ^= unsafe { *i.add(a.trailing_zeros() as usize) };

        // Clear the least significant 1 bit
        a &= a - 1;
    }

    Float::min(randomizer.scramble(v) as Float * hexf32!("0x1.0p-32") as Float, ONE_MINUS_EPSILON)
}

#[inline]
pub fn compute_radical_inverse_permutations(seed: i32) -> Vec<DigitPermutation> {
    let mut perms = Vec::with_capacity(PRIME_TABLE_SIZE as usize);
    for p in PRIMES.iter() {
        perms.push(DigitPermutation::new(*p, seed));
    }
    perms
}

#[inline]
pub fn radical_inverse(base_index: u32, mut a: u64) -> Float {
    let base = PRIMES[base_index as usize] as u64;

    let limit = (!0u64) / base - base;
    let inv_base = 1.0 / base as Float;
    let mut inv_base_m = 1.0;
    let mut reversed_digits = 0u64;

    while a != 0 && reversed_digits < limit {
        let next = a / base;
        let digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_m *= inv_base;
        a = next;
    }

    Float::min(reversed_digits as Float * inv_base_m, ONE_MINUS_EPSILON)
}

#[inline]
pub fn inverse_radical_inverse(mut inverse: u64, base: u64, n_digits: i32) -> u64 {
    let mut index = 0;

    for i in 0..n_digits {
        let digit = inverse % base;
        inverse /= base;
        index = index * base + digit;
    }

    index
}

pub fn scrambled_radical_inverse(base_index: u32, mut a: u64, perm: &DigitPermutation) -> Float {
    let base = PRIMES[base_index as usize] as u64;
    let limit = (!0u64) / base - base;
    let inv_base = 1.0 / base as Float;
    let mut inv_base_m = 1.0;
    let mut reversed_digits = 0u64;
    let mut digit_index = 0;

    while 1.0 - (base - 1) as Float * inv_base_m < 1.0 && reversed_digits < limit {
        let next = a / base;
        let digit_value = a - next * base;
        reversed_digits = reversed_digits * base + perm.permute(digit_index, digit_value as u32) as u64;
        inv_base_m *= inv_base;
        digit_index += 1;
        a = next;
    }

    Float::min(inv_base_m * reversed_digits as Float, ONE_MINUS_EPSILON)
}

pub fn owen_scrambled_radical_inverse(base_index: u32, mut a: u64, hash: u64) -> Float {
    let base = PRIMES[base_index as usize] as u64;
    let limit = (!0u64) / base - base;
    let inv_base = 1.0 / base as Float;
    let mut inv_base_m = 1.0;
    let mut reversed_digits = 0u64;
    let mut digit_index = 0;

    while 1.0 - inv_base_m < 1.0 && reversed_digits < limit {
        let next = a / base;
        let mut digit_value = a - next * base;
        let digit_hash = mix_bits(hash ^ reversed_digits) as u32;
        digit_value = permutation_element(digit_value as u32, base as u32, digit_hash) as u64;;
        reversed_digits = reversed_digits * base + digit_value;
        inv_base_m *= inv_base;
        digit_index += 1;
        a = next;
    }

    Float::min(inv_base_m * reversed_digits as Float, ONE_MINUS_EPSILON)
}
