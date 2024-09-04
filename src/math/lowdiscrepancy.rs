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

// TODO: Determine whether tail recursion or the while loop is faster
#[inline(always)]
fn sobol_reduce_randomize(a: u64, i: usize, v: u32) -> u32 {
    match a {
        0 => v,
        _ => sobol_reduce_randomize(a >> 1, i + 1, if a & 1 != 0 {
            unsafe { v ^ *SOBOL_MATRICES_32.get_unchecked(i) }
        } else { v }) ,
    }
}

#[inline]
pub fn sobol_sample<R: AbstractScrambler>(mut a: u64, dimension: usize, randomizer: R) -> Float {
    debug_assert!(dimension < N_SOBOL_DIMENSIONS as usize);
    debug_assert!(a < (1u64 << SOBOL_MATRIX_SIZE));

    let mut v = 0u32;
    let mut i = dimension * SOBOL_MATRIX_SIZE as usize;
    // let v = sobol_reduce_randomize(a, i, 0);
    
    // TODO: Improve this algorithm by using trailing_zeros()
    while a != 0 {
        if a & 1 != 0 {
            v ^= *unsafe { SOBOL_MATRICES_32.get_unchecked(i) };
        }

        a >>= 1;
        i += 1;
    }

    Float::min(randomizer.scramble(v) as Float * hexf32!("0x1.0p-32") as Float, ONE_MINUS_EPSILON)
}
