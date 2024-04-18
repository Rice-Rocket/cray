use crate::math::*;

macro_rules! create_bivec {
    ($bivec:ident, $vec:ident) => {
        #[derive(Clone, Copy, PartialEq, Debug)]
        pub struct $bivec(pub Unit<$vec>);

        impl $bivec {
            #[doc = concat!("Creates a new [`", stringify!($bivec), "`] given a normalized [`", stringify!($vec), "`].")]
            #[inline]
            pub const fn new(v: $vec) -> Self {
                Self(Unit::new_unchecked(v))
            }

            #[doc = concat!("Creates a new [`", stringify!($bivec), "`], normalizing the given [`", stringify!($vec), "`] in the process.")]
            #[inline]
            pub fn new_normalize(v: $vec) -> Self {
                Self(Unit::new_normalize(v))
            }

            #[doc = concat!("Returns the inner [`", stringify!($vec), "`], consuming `self`.")]
            #[inline]
            pub fn inner(self) -> $vec {
                self.0.into_inner()
            }

            #[doc = concat!("Returns a reference to the inner [`", stringify!($vec), "`].")]
            #[inline]
            pub fn get(&self) -> &$vec {
                self.0.as_ref()
            }

            #[inline]
            pub fn dot(&self, rhs: &Self) -> Scalar {
                self.get().dot(rhs.get())
            }
            
            #[inline]
            pub fn angle(&self, rhs: &Self) -> Scalar {
                self.get().angle(rhs.get())
            }

            #[inline]
            pub fn x(&self) -> Scalar {
                self.get().x
            }

            #[inline]
            pub fn y(&self) -> Scalar {
                self.get().y
            }
        }
    }
}

create_bivec!(Bivec2, Vec2);
create_bivec!(Bivec3, Vec3);
create_bivec!(Bivec4, Vec4);


impl Bivec3 {
    #[inline]
    pub fn z(&self) -> Scalar {
        self.get().z
    }

    #[inline]
    pub fn cross(&self, rhs: &Self) -> Vec3 {
        self.get().cross(rhs.get())
    }
}

impl Bivec4 {
    #[inline]
    pub fn z(&self) -> Scalar {
        self.get().z
    }

    #[inline]
    pub fn w(&self) -> Scalar {
        self.get().w
    }
}
