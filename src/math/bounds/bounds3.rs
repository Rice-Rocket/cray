// Pbrt 3.7 Bounding Boxes

use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, Index, Mul, Sub};

use bounds::Union;

use crate::math::*;

/// A 3-dimensional axis-aligned bounding box of type `T`
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Bounds3<T: Clone + Copy> {
    pub min: Point3<T>,
    pub max: Point3<T>,
}

impl<T> Bounds3<T>
where
    T: NumericConsts + NumericOrd + PartialOrd + Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>
{
    /// Creates a new [`TBounds3`] with from a minimum and maximum.
    #[inline]
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        Self { min: p1.min(p2), max: p1.max(p2) }
    }

    /// Creates a new [`TBounds3`] containing a single `point`.
    #[inline]
    pub const fn from_point(point: Point3<T>) -> Self {
        Self { min: point, max: point }
    }

    /// Creates a new [`TBounds3`] containing the given `points`.
    #[inline]
    pub fn from_points(points: Vec<Point3<T>>) -> Self {
        points.iter().fold(Bounds3::default(), |bounds, p| bounds.union(*p))
    }

    /// Returns the position of the given `corner`.
    #[inline]
    pub fn corner(self, corner: u8) -> Point3<T> {
        Point3::new(self[corner & 1].x, self[if corner & 2 != 0 { 1 } else { 0 }].y, self[if corner & 4 != 0 { 1 } else { 0 }].z)
    }

    /// Returns the diagonal vector of the bounding box.
    ///
    /// In other words, the vector that points from `self.min` to `self.max`.
    #[inline]
    #[doc(alias = "dimensions")]
    pub fn diagonal(self) -> Point3<T> {
        self.max - self.min
    }

    /// Returns the dimensions of the bounding box.
    #[inline(always)]
    #[doc(alias = "diagonal")]
    pub fn dimensions(self) -> Point3<T> {
        self.diagonal()
    }

    /// Computes the surface area of the bounding box.
    #[inline]
    pub fn surface_area(self) -> T {
        let d = self.diagonal();
        T::TWO * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    /// Computes the volume of the bounding box.
    #[inline]
    pub fn volume(self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    /// Computes the longest dimension of the bounding box.
    #[inline]
    pub fn max_dim(self) -> Dimension {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            Dimension::X
        } else if d.y > d.z {
            Dimension::Y
        } else {
            Dimension::Z
        }
    }

    /// Lerps between the `min` and `max` of this [`TBounds3`] based on `t`.
    ///
    /// This essentially allows you to select an arbitrary point inside the
    /// bounding box.
    #[inline]
    pub fn lerp(self, t: Point3<T>) -> Point3<T> {
        Point3::new(
            lerp(self.min.x, self.max.x, t.x),
            lerp(self.min.y, self.max.y, t.y),
            lerp(self.min.z, self.max.z, t.z),
        )
    }

    /// Computes the position of a point `p` relative to the corners of the box.
    ///
    /// That is, if `p = self.min`, the offset is `(0, 0, 0)`. If `p =
    /// self.max`, the offset is `(1, 1, 1)` and so on for values in between.
    #[inline]
    pub fn offset(self, p: Point3<T>) -> Point3<T> {
        let mut o = p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / (self.max.x - self.min.x)
        };
        if self.max.y > self.min.y {
            o.y = o.y / (self.max.y - self.min.y)
        };
        if self.max.z > self.min.z {
            o.z = o.z / (self.max.z - self.min.z)
        };
        o
    }

    /// Returns whether or not this bounding box is empty (whether it has no
    /// points in it).
    #[inline]
    pub fn is_empty(self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y || self.min.z >= self.max.z
    }

    /// Returns whether or not this bounding box is degenerate.
    #[inline]
    pub fn is_degenerate(self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Computes whether or not the given point `p` is inside the bounding box.
    #[inline]
    pub fn inside(self, p: Point3<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Same as `inside`, but excludes points on the upper boundary.
    #[inline]
    pub fn inside_exclusive(self, p: Point3<T>) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y && p.z >= self.min.z && p.z < self.max.z
    }

    /// Computes whether or not two bounding boxes overlap.
    #[inline]
    pub fn overlaps(self, rhs: Self) -> bool {
        ((self.max.x >= rhs.min.x) && (self.min.x <= rhs.max.x))
            && ((self.max.y >= rhs.min.y) && (self.min.y <= rhs.max.y))
            && ((self.max.z >= rhs.min.z) && (self.min.z <= rhs.max.z))
    }

    /// Computes the squared distance from point `p` to the edge of the bounding
    /// box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance_sqr(self, p: Point3<T>) -> T {
        let dx = T::ZERO.nmax(self.min.x - p.x).nmax(p.x - self.max.x);
        let dy = T::ZERO.nmax(self.min.y - p.y).nmax(p.y - self.max.y);
        let dz = T::ZERO.nmax(self.min.z - p.z).nmax(p.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }

    /// Pads the bounding box by a constant amount `delta` equally in all
    /// dimensions.
    #[inline]
    pub fn expand(self, delta: T) -> Self {
        Self { min: self.min - Point3::new(delta, delta, delta), max: self.max + Point3::new(delta, delta, delta) }
    }
}

impl Bounds3<Float> {
    pub fn intersect_p(&self, o: Point3f, d: Vec3f, t_max: Float) -> Option<HitTimes> {
        let mut t0 = 0.0;
        let mut t1 = t_max;
        for i in 0..3 {
            let inv_ray_dir = 1.0 / d[i];
            let t_near = (self.min[i] - o[i]) * inv_ray_dir;
            let t_far = (self.max[i] - o[i]) * inv_ray_dir;
            let (t_near, t_far) = if t_near > t_far {
                (t_far, t_near)
            } else {
                (t_near, t_far)
            };

            let t_far = t_far * (1.0 + 2.0 * gamma(3));

            t0 = if t_near > 0.0 { t_near } else { t0 };
            t1 = if t_far < t1 { t_far } else { t1 };
            if t0 > t1 {
                return None;
            }
        }

        Some(HitTimes { t0, t1 })
    }

    pub fn intersect_p_cached(&self, o: Point3f, _d: Vec3f, ray_t_max: Float, inv_dir: Vec3f, dir_is_neg: [u8; 3]) -> bool {
        let mut t_min = (self[dir_is_neg[0]].x - o.x) * inv_dir.x;
        let mut t_max = (self[1 - dir_is_neg[0]].x - o.x) * inv_dir.x;
        let ty_min = (self[dir_is_neg[1]].y - o.y) * inv_dir.y;
        let mut ty_max = (self[1 - dir_is_neg[1]].y - o.y) * inv_dir.y;

        t_max *= 1.0 + 2.0 * gamma(3);
        ty_max *= 1.0 + 2.0 * gamma(3);

        if t_min > ty_max || ty_min > t_max {
            return false;
        }
        if ty_min > t_min {
            t_min = ty_min;
        }
        if ty_max < t_max {
            t_max = ty_max;
        }

        // Check for ray intersection
        let tz_min = (self[dir_is_neg[2]].z - o.z) * inv_dir.z;
        let mut tz_max = (self[1 - dir_is_neg[2]].z - o.z) * inv_dir.z;
        // Update the maximum value to ensure robust bounds intersection
        tz_max *= 1.0 + 2.0 * gamma(3);

        if t_min > tz_max || tz_min > t_max {
            return false;
        }
        if tz_min > t_min {
            t_min = tz_min;
        }
        if tz_max < t_max {
            t_max = tz_max;
        }

        t_min < ray_t_max && t_max > 0.0
    }
}

pub struct HitTimes {
    pub t0: Float,
    pub t1: Float,
}

impl<T> Default for Bounds3<T>
where
    T: NumericConsts + NumericOrd + PartialOrd + Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>
{
    /// Creates a new [`TBounds3`] with no points.
    fn default() -> Self {
        Self::new(Point3::MAX, Point3::MIN)
    }
}

impl<T> Bounds3<T>
where
    T: NumericField + NumericNegative + NumericFloat + NumericOrd
{
    /// Computes the bounding sphere that encompasses this bounding box.
    #[inline]
    pub fn bounding_sphere(self) -> (Point3<T>, T) {
        let center = (self.min + self.max) / T::TWO;
        let radius = if self.inside(center) { (center - self.max).length() } else { T::ZERO };
        (center, radius)
    }

    /// Computes the distance from point `p` to the edge of the bounding box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance(self, p: Point3<T>) -> T {
        self.distance_sqr(p).nsqrt()
    }
}


impl<T: NumericOrd + Clone + Copy> Bounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn union_box(self, rhs: Self) -> Self {
        Self {
            min: self.min.min(rhs.min),
            max: self.max.max(rhs.max),
        }
    }

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn union_vect(self, rhs: Point3<T>) -> Self {
        Self {
            min: self.min.min(rhs),
            max: self.max.max(rhs),
        }
    }

    /// Takes the intersection of two bounding boxes
    #[inline]
    pub fn intersect(self, rhs: Self) -> Self {
        Self {
            min: Point3::new(self.min.x.nmax(rhs.min.x), self.min.y.nmax(rhs.min.y), self.min.z.nmax(rhs.min.z)),
            max: Point3::new(self.max.x.nmin(rhs.max.x), self.max.y.nmin(rhs.max.y), self.max.z.nmin(rhs.max.z)),
        }
    }
}


impl<T: Clone + Copy> Index<u8> for Bounds3<T> {
    type Output = Point3<T>;

    /// Indexes into the extremes of the bounding box.
    ///
    /// An index of `0` returns `self.min` while an index of `1` returns
    /// `self.max`.
    #[inline]
    fn index(&self, index: u8) -> &Self::Output {
        if index == 0 {
            &self.min
        } else {
            &self.max
        }
    }
}

impl<T: NumericOrd + Clone + Copy> Union<Bounds3<T>> for Bounds3<T> {
    type Output = Bounds3<T>;

    fn union(self, rhs: Bounds3<T>) -> Self::Output {
        self.union_box(rhs)
    }
}

impl<T: NumericOrd + Clone + Copy> Union<Point3<T>> for Bounds3<T> {
    type Output = Bounds3<T>;

    fn union(self, rhs: Point3<T>) -> Self::Output {
        self.union_vect(rhs)
    }
}
