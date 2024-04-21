use crate::math::*;

pub struct Frame {
    pub x: UnitVec3f,
    pub y: UnitVec3f,
    pub z: UnitVec3f,
}

impl Frame {
    #[inline]
    pub fn new(x: UnitVec3f, y: UnitVec3f, z: UnitVec3f) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn from_xz(x: UnitVec3f, z: UnitVec3f) -> Self {
        Self::new(x, z.cross(x), z)
    }

    #[inline]
    pub fn from_xy(x: UnitVec3f, y: UnitVec3f) -> Self {
        Self::new(x, y, x.cross(y))
    }

    #[inline]
    pub fn from_yz(y: UnitVec3f, z: UnitVec3f) -> Self {
        Self::new(y.cross(z), y, z)
    }

    #[inline]
    pub fn from_x(x: UnitVec3f) -> Self {
        let bases = x.local_basis();
        Self::new(x, bases.0, bases.1)
    }

    #[inline]
    pub fn from_y(y: UnitVec3f) -> Self {
        let bases = y.local_basis();
        Self::new(bases.1, y, bases.0)
    }
    
    #[inline]
    pub fn from_z(z: UnitVec3f) -> Self {
        let bases = z.local_basis();
        Self::new(bases.0, bases.1, z)
    }

    #[inline]
    pub fn localize(self, v: UnitVec3f) -> UnitVec3f {
        UnitVec3f::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }
    
    #[inline]
    pub fn globalize(self, v: UnitVec3f) -> UnitVec3f {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}
