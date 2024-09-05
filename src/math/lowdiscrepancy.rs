use hexf::hexf32;

use crate::{hashing::mix_bits, sobol::{N_SOBOL_DIMENSIONS, SOBOL_MATRICES_32, SOBOL_MATRIX_SIZE}, Float, ONE_MINUS_EPSILON};

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
