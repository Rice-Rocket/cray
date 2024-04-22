use crate::math::*;

pub struct Frame {
    pub x: Vec3f,
    pub y: Vec3f,
    pub z: Vec3f,
}

impl Frame {
    #[inline]
    pub fn new(x: Vec3f, y: Vec3f, z: Vec3f) -> Self {
        math_assert!(x.is_normalized());
        math_assert!(y.is_normalized());
        math_assert!(z.is_normalized());

        Self { x, y, z }
    }

    #[inline]
    pub fn from_xz(x: Vec3f, z: Vec3f) -> Self {
        math_assert!(x.is_normalized());
        math_assert!(z.is_normalized());

        Self::new(x, z.cross(x), z)
    }

    #[inline]
    pub fn from_xy(x: Vec3f, y: Vec3f) -> Self {
        math_assert!(x.is_normalized());
        math_assert!(y.is_normalized());

        Self::new(x, y, x.cross(y))
    }

    #[inline]
    pub fn from_yz(y: Vec3f, z: Vec3f) -> Self {
        math_assert!(y.is_normalized());
        math_assert!(z.is_normalized());

        Self::new(y.cross(z), y, z)
    }

    #[inline]
    pub fn from_x(x: Vec3f) -> Self {
        math_assert!(x.is_normalized());

        let bases = x.local_basis();
        Self::new(x, bases.0, bases.1)
    }

    #[inline]
    pub fn from_y(y: Vec3f) -> Self {
        math_assert!(y.is_normalized());

        let bases = y.local_basis();
        Self::new(bases.1, y, bases.0)
    }
    
    #[inline]
    pub fn from_z(z: Vec3f) -> Self {
        math_assert!(z.is_normalized());

        let bases = z.local_basis();
        Self::new(bases.0, bases.1, z)
    }

    #[inline]
    pub fn localize(self, v: Vec3f) -> Vec3f {
        Vec3f::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }
    
    #[inline]
    pub fn globalize(self, v: Vec3f) -> Vec3f {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}
