// Pbrt 3.7 Bounding Boxes

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Index};

use crate::math::*;

/// A 3-dimensional axis-aligned bounding box of type `T`
// TODO: Test this
pub struct TBounds3<T> {
    pub min: TVec3<T>,
    pub max: TVec3<T>,
}

impl<T> TBounds3<T>
where
    T: Numeric + PartialOrd + Clone + Copy,
{
    /// Creates a new [`TBounds3`] with no points.
    #[inline]
    pub const fn new() -> Self {
        Self { min: TVec3::new(T::MAX, T::MAX, T::MAX), max: TVec3::new(T::MIN, T::MIN, T::MIN) }
    }

    /// Creates a new [`TBounds3`] containing a single `point`.
    #[inline]
    pub const fn from_point(point: TVec3<T>) -> Self {
        Self { min: point, max: point }
    }

    /// Creates a new [`TBounds3`] containing the given `points`.
    #[inline]
    pub fn from_points(points: Vec<TVec3<T>>) -> Self {
        points.iter().fold(TBounds3::new(), |bounds, p| bounds | p)
    }

    /// Returns the position of the given `corner`.
    #[inline]
    pub fn corner(&self, corner: u8) -> TVec3<T> {
        TVec3::new(self[corner & 1].x, self[if corner & 2 != 0 { 1 } else { 0 }].y, self[if corner & 4 != 0 { 1 } else { 0 }].z)
    }

    /// Returns the diagonal vector of the bounding box.
    ///
    /// In other words, the vector that points from `self.min` to `self.max`.
    #[inline]
    #[doc(alias = "dimensions")]
    pub fn diagonal(&self) -> TVec3<T> {
        self.max - self.min
    }

    /// Returns the dimensions of the bounding box.
    #[inline(always)]
    #[doc(alias = "diagonal")]
    pub fn dimensions(&self) -> TVec3<T> {
        self.diagonal()
    }

    /// Computes the surface area of the bounding box.
    #[inline]
    pub fn surface_area(&self) -> T {
        let d = self.diagonal();
        T::TWO * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    /// Computes the volume of the bounding box.
    #[inline]
    pub fn volume(&self) -> T {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    /// Computes the longest dimension of the bounding box.
    #[inline]
    pub fn max_dim(&self) -> Dimension {
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
    pub fn lerp(&self, t: &TVec3<T>) -> TVec3<T> {
        TVec3::new(
            math::lerp(self.min.x, self.max.x, t.x),
            math::lerp(self.min.y, self.max.y, t.y),
            math::lerp(self.min.z, self.max.z, t.z),
        )
    }

    /// Computes the position of a point `p` relative to the corners of the box.
    ///
    /// That is, if `p = self.min`, the offset is `(0, 0, 0)`. If `p =
    /// self.max`, the offset is `(1, 1, 1)` and so on for values in between.
    #[inline]
    pub fn offset(&self, p: &TVec3<T>) -> TVec3<T> {
        let mut o = p - self.min;
        if self.max.x > self.min.x {
            o.x /= self.max.x - self.min.x
        };
        if self.max.y > self.min.y {
            o.y /= self.max.y - self.min.y
        };
        if self.max.z > self.min.z {
            o.z /= self.max.z - self.min.z
        };
        o
    }

    /// Computes the bounding sphere that encompasses this bounding box.
    #[inline]
    pub fn bounding_sphere(&self) -> (TVec3<T>, T) {
        let center = (self.min + self.max) / T::TWO;
        let radius = if self.inside(&center) { (center - self.max).magnitude() } else { T::ZERO };
        (center, radius)
    }

    /// Returns whether or not this bounding box is empty (whether it has no
    /// points in it).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y || self.min.z >= self.max.z
    }

    /// Returns whether or not this bounding box is degenerate.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Computes whether or not the given point `p` is inside the bounding box.
    #[inline]
    pub fn inside(&self, p: &TVec3<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Same as `inside`, but excludes points on the upper boundary.
    #[inline]
    pub fn inside_exclusive(&self, p: &TVec3<T>) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y && p.z >= self.min.z && p.z < self.max.z
    }

    /// Computes whether or not two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, rhs: &Self) -> bool {
        ((self.max.x >= rhs.min.x) && (self.min.x <= rhs.max.x))
            && ((self.max.y >= rhs.min.y) && (self.min.y <= rhs.max.y))
            && ((self.max.z >= rhs.min.z) && (self.min.z <= rhs.max.z))
    }

    /// Computes the squared distance from point `p` to the edge of the bounding
    /// box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance_sqr(&self, p: &TVec3<T>) -> T {
        let dx = T::ZERO.maxi(self.min.x - p.x).maxi(p.x - self.max.x);
        let dy = T::ZERO.maxi(self.min.y - p.y).maxi(p.y - self.max.y);
        let dz = T::ZERO.maxi(self.min.z - p.z).maxi(p.z - self.max.z);
        dx * dx + dy * dy + dz * dz
    }

    /// Computes the distance from point `p` to the edge of the bounding box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance(&self, p: &TVec3<T>) -> T {
        self.distance_sqr(p).sqrt()
    }

    /// Pads the bounding box by a constant amount `delta` equally in all
    /// dimensions.
    #[inline]
    pub fn expand(self, delta: T) -> Self {
        Self { min: self.min - TVec3::new(delta, delta, delta), max: self.max + TVec3::new(delta, delta, delta) }
    }
}


impl<T: Numeric + Clone + Copy> TBounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[doc(alias = "|, |=")]
    #[inline]
    pub fn union_box(self, rhs: &Self) -> Self {
        Self {
            min: TVec3::new(self.min.x.mini(rhs.min.x), self.min.y.mini(rhs.min.y), self.min.z.mini(rhs.min.z)),
            max: TVec3::new(self.max.x.maxi(rhs.max.x), self.max.y.maxi(rhs.max.y), self.max.z.maxi(rhs.max.z)),
        }
    }

    fn union_box_assign(&mut self, rhs: &Self) {
        self.min = TVec3::new(self.min.x.mini(rhs.min.x), self.min.y.mini(rhs.min.y), self.min.z.mini(rhs.min.z));
        self.max = TVec3::new(self.max.x.maxi(rhs.max.x), self.max.y.maxi(rhs.max.y), self.max.z.maxi(rhs.max.z));
    }

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[doc(alias = "|, |=")]
    #[inline]
    pub fn union_vect(self, rhs: &TVec3<T>) -> Self {
        Self {
            min: TVec3::new(self.min.x.mini(rhs.x), self.min.y.mini(rhs.y), self.min.z.mini(rhs.z)),
            max: TVec3::new(self.max.x.maxi(rhs.x), self.max.y.maxi(rhs.y), self.max.z.maxi(rhs.z)),
        }
    }

    fn union_vect_assign(&mut self, rhs: &TVec3<T>) {
        self.min = TVec3::new(self.min.x.mini(rhs.x), self.min.y.mini(rhs.y), self.min.z.mini(rhs.z));
        self.max = TVec3::new(self.max.x.maxi(rhs.x), self.max.y.maxi(rhs.y), self.max.z.maxi(rhs.z));
    }

    /// Takes the intersection of two bounding boxes
    #[doc(alias = "&, &=")]
    #[inline]
    pub fn intersect(self, rhs: &Self) -> Self {
        Self {
            min: TVec3::new(self.min.x.maxi(rhs.min.x), self.min.y.maxi(rhs.min.y), self.min.z.maxi(rhs.min.z)),
            max: TVec3::new(self.max.x.mini(rhs.max.x), self.max.y.mini(rhs.max.y), self.max.z.mini(rhs.max.z)),
        }
    }

    fn intersect_assign(&mut self, rhs: &Self) {
        self.min = TVec3::new(self.min.x.maxi(rhs.min.x), self.min.y.maxi(rhs.min.y), self.min.z.maxi(rhs.min.z));
        self.max = TVec3::new(self.max.x.mini(rhs.max.x), self.max.y.mini(rhs.max.y), self.max.z.mini(rhs.max.z));
    }
}


impl<T> Index<u8> for TBounds3<T> {
    type Output = TVec3<T>;

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


impl<T: Numeric + Clone + Copy> BitOr<&TBounds3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor(self, rhs: &TBounds3<T>) -> Self::Output {
        self.union_box(rhs)
    }
}


impl<T: Numeric + Clone + Copy> BitOr<TBounds3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor(self, rhs: TBounds3<T>) -> Self::Output {
        self.union_box(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOr<TVec3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor(self, rhs: TVec3<T>) -> Self::Output {
        self.union_vect(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOr<&TVec3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor(self, rhs: &TVec3<T>) -> Self::Output {
        self.union_vect(rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitAnd<&TBounds3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand(self, rhs: &TBounds3<T>) -> Self::Output {
        self.intersect(rhs)
    }
}


impl<T: Numeric + Clone + Copy> BitAnd<TBounds3<T>> for TBounds3<T> {
    type Output = TBounds3<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand(self, rhs: TBounds3<T>) -> Self::Output {
        self.intersect(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<TBounds3<T>> for TBounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: TBounds3<T>) {
        self.union_box_assign(&rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<&TBounds3<T>> for TBounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: &TBounds3<T>) {
        self.union_box_assign(rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<TVec3<T>> for TBounds3<T> {
    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: TVec3<T>) {
        self.union_vect_assign(&rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<&TVec3<T>> for TBounds3<T> {
    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: &TVec3<T>) {
        self.union_vect_assign(rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitAndAssign<&TBounds3<T>> for TBounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand_assign(&mut self, rhs: &TBounds3<T>) {
        self.intersect_assign(rhs);
    }
}


impl<T: Numeric + Clone + Copy> BitAndAssign<TBounds3<T>> for TBounds3<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand_assign(&mut self, rhs: TBounds3<T>) {
        self.intersect_assign(&rhs);
    }
}
