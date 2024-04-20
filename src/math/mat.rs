use std::ops::{Index, IndexMut};

macro_rules! create_mat {
    ($name:ident; $row:expr, $col:expr; $($($m:ident)-*),* $(;)*) => {
        pub struct $name<T>([[T; $col]; $row]);

        impl<T> $name<T> {
            pub fn new($($($m: T,)*)*) -> Self {
                Self([$([$($m,)*],)*])
            }
        }

        impl<T> Index<(usize, usize)> for $name<T> {
            type Output = T;

            fn index(&self, index: (usize, usize)) -> &Self::Output {
                &self.0[index.0][index.1]
            }
        }

        impl<T> IndexMut<(usize, usize)> for $name<T> {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                &mut self.0[index.0][index.1]
            }
        }
    }
}


create_mat!(TMat2; 2, 2; m00-m01,m10-m11);
create_mat!(TMat3; 3, 3; m00-m01-m02,m10-m11-m12,m20-m21-m22);
create_mat!(TMat4; 4, 4; m00-m01-m02-m03,m10-m11-m12-m13,m20-m21-m22-m23,m30-m31-m32-m33);
