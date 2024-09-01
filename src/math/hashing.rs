#[inline]
pub fn mix_bits(mut v: u64) -> u64 {
    v ^= v.wrapping_shr(31);
    v = v.wrapping_mul(0x7fb5d329728ea185);
    v ^= v.wrapping_shr(27);
    v = v.wrapping_mul(0x81dadef4bc2dd44d);
    v ^= v.wrapping_shr(33);
    v
}

pub fn murmur_hash_64a(key: &[u8], seed: u64) -> u64 {
    const M: u64 = 0xc6a4a7935bd1e995;
    const R: u8 = 47;

    let len = key.len();
    let mut h = seed ^ (len as u64).wrapping_mul(M);

    let endpos = len - (len & 7);
    let mut i = 0;

    while i != endpos {
        let mut k: u64;

        k = key[i] as u64;
        k |= (key[i + 1] as u64) << 8;
        k |= (key[i + 2] as u64) << 16;
        k |= (key[i + 3] as u64) << 24;
        k |= (key[i + 4] as u64) << 32;
        k |= (key[i + 5] as u64) << 40;
        k |= (key[i + 6] as u64) << 48;
        k |= (key[i + 7] as u64) << 56;

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);

        i += 8;
    }

    let over = len & 7;
    if over == 7 { h ^= (key[i + 6] as u64) << 48; }
    if over >= 6 { h ^= (key[i + 5] as u64) << 40; }
    if over >= 5 { h ^= (key[i + 4] as u64) << 32; }
    if over >= 4 { h ^= (key[i + 3] as u64) << 24; }
    if over >= 3 { h ^= (key[i + 2] as u64) << 16; }
    if over >= 2 { h ^= (key[i + 1] as u64) << 8; }
    if over >= 1 { h ^= key[i] as u64; }
    if over > 0 { h = h.wrapping_mul(M); }

    h ^= h >> R;
    h = h.wrapping_mul(M);
    h ^= h >> R;
    h
}

#[macro_export]
macro_rules! hash {
    ($($arg:expr),*) => {
        {
            $crate::math::hashing::murmur_hash_64a(&hash!(@genkey $($arg),*), 0)
        }
    };
    (@genkey $($arg:expr),*) => {
        {
            let mut key = Vec::new();
            $(
                key.extend_from_slice(&$arg.to_be_bytes());
            )*
            key
        }
    };
}

#[macro_export]
macro_rules! transmute {
    ($v:expr => $to:ident) => {
        {
            unsafe {
                std::mem::transmute_copy::<_, $to>(&$v)
            }
        }
    };
}

const C1: u32 = 0xcc9e2d51;
const C2: u32 = 0x1b873593;

#[inline]
pub fn split_u64(v: u64) -> (u32, u32) {
    ((v & 0xffffffff) as u32, v.wrapping_shr(32) as u32)
}

#[inline]
pub fn combine_low_bits_32(a: u32, b: u32) -> u32 {
    a ^ (b.wrapping_shl(16))
}

#[inline]
pub fn combine_bits_64(a: u64, b: u64) -> u64 {
    a ^ (b.wrapping_shl(32))
}

#[inline]
fn rotl(x: u32, r: u32) -> u32 {
    (x.wrapping_shl(r)) | (x.wrapping_shr(32u32.wrapping_sub(r)))
}

#[inline]
fn rotr(x: u32, r: u32) -> u32 {
    (x.wrapping_shr(r)) | (x.wrapping_shl(32u32.wrapping_sub(r)))
}

/// Hash a single 32-bit value
#[inline]
pub fn fmix(mut h: u32) -> u32 {
    h ^= h.wrapping_shr(16);
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h.wrapping_shr(13);
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h.wrapping_shr(16);
    h
}

/// Combine two 32-bit values
#[inline]
pub fn mur(mut a: u32, mut b: u32) -> u32 {
    a = a.wrapping_mul(C1);
    a = rotr(a, 17);
    a = a.wrapping_mul(C2);
    b ^= a;
    b = rotr(b, 19);
    b.wrapping_mul(5).wrapping_add(0xe6546b64)
}

#[inline]
fn bswap32(x: u32) -> u32 {
    ((x & 0x000000ff).wrapping_shl(24)) |
        ((x & 0x0000ff00).wrapping_shl(8)) |
        ((x & 0x00ff0000).wrapping_shr(8)) |
        ((x & 0xff000000).wrapping_shr(24))
}

#[inline]
pub fn ihash21(mut x: u32, mut y: u32) -> u32 {
    x = x.wrapping_mul(73333);
    y = y.wrapping_mul(7777);
    x ^= 3333777777u32.wrapping_shr(x.wrapping_shr(28));
    y ^= 3333777777u32.wrapping_shr(y.wrapping_shr(28));
    let n = x.wrapping_mul(y);
    n ^ (n.wrapping_shr(15))
}

#[inline]
pub fn ihash31(mut x: u32, mut y: u32, mut z: u32) -> u32 {
    const LEN: u32 = 12;
    let a = LEN + bswap32(x);
    let b = LEN * 5 + bswap32(z);
    let c = 9 + bswap32(y);

    fmix(mur(c, mur(b, mur(a, b))))
}

#[inline]
pub fn ihash41(mut x: u32, mut y: u32, mut z: u32, mut w: u32) -> u32 {
    const LEN: u32 = 16;
    let a = bswap32(w);
    let b = bswap32(y);
    let c = bswap32(z);
    let e = bswap32(x);

    fmix(mur(a, mur(e, mur(c, mur(c, mur(b, mur(a, LEN)))))))
}

#[inline]
pub fn permutation_element(mut i: u32, l: u32, p: u32) -> i32 {
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

pub fn left_shift_2(mut x: u64) -> u64 {
    x &= 0xffffffff;
    x = (x ^ (x.wrapping_shl(16))) & 0x0000ffff0000ffff;
    x = (x ^ (x.wrapping_shl(8))) & 0x00ff00ff00ff00ff;
    x = (x ^ (x.wrapping_shl(4))) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x.wrapping_shl(2))) & 0x3333333333333333;
    x = (x ^ (x.wrapping_shl(1))) & 0x5555555555555555;
    x
}

pub fn encode_morton_2(x: u32, y: u32) -> u64 {
    (left_shift_2(y as u64) << 1) | left_shift_2(x as u64)
}
