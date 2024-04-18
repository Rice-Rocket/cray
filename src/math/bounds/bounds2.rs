// Pbrt 3.7 Bounding Boxes

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Index};

use crate::math::*;

/// A 2-dimensional axis-aligned bounding box of type `T`
// TODO: Test this
pub struct TBounds2<T> {
    pub min: TVec2<T>,
    pub max: TVec2<T>,
}

impl<T> TBounds2<T>
where
    T: Numeric + PartialOrd + Clone + Copy,
{
    /// Creates a new [`TBounds2`] with no points.
    #[inline]
    pub const fn new() -> Self {
        Self { min: TVec2::new(T::MAX, T::MAX), max: TVec2::new(T::MIN, T::MIN) }
    }

    /// Creates a new [`TBounds2`] containing a single `point`.
    #[inline]
    pub const fn from_point(point: TVec2<T>) -> Self {
        Self { min: point, max: point }
    }

    /// Creates a new [`TBounds2`] containing the given `points`.
    #[inline]
    pub fn from_points(points: Vec<TVec2<T>>) -> Self {
        points.iter().fold(TBounds2::new(), |bounds, p| bounds | p)
    }

    /// Returns the position of the given `corner`.
    #[inline]
    pub fn corner(&self, corner: u8) -> TVec2<T> {
        TVec2::new(self[corner & 1].x, self[if corner & 2 != 0 { 1 } else { 0 }].y)
    }

    /// Returns the diagonal vector of the bounding box.
    ///
    /// In other words, the vector that points from `self.min` to `self.max`.
    #[inline]
    #[doc(alias = "dimensions")]
    pub fn diagonal(&self) -> TVec2<T> {
        self.max - self.min
    }

    /// Returns the dimensions of the bounding box.
    #[inline(always)]
    #[doc(alias = "diagonal")]
    pub fn dimensions(&self) -> TVec2<T> {
        self.diagonal()
    }

    /// Computes the area of the bounding box.
    #[inline]
    pub fn area(&self) -> T {
        let d = self.diagonal();
        d.x * d.y
    }

    /// Computes the longest dimension of the bounding box.
    #[inline]
    pub fn max_dim(&self) -> Dimension {
        let d = self.diagonal();
        if d.x > d.y {
            Dimension::X
        } else {
            Dimension::Y
        }
    }

    /// Lerps between the `min` and `max` of this [`TBounds2`] based on `t`.
    ///
    /// This essentially allows you to select an arbitrary point inside the
    /// bounding box.
    #[inline]
    pub fn lerp(&self, t: &TVec2<T>) -> TVec2<T> {
        TVec2::new(math::lerp(self.min.x, self.max.x, t.x), math::lerp(self.min.y, self.max.y, t.y))
    }

    /// Computes the position of a point `p` relative to the corners of the box.
    ///
    /// That is, if `p = self.min`, the offset is `(0, 0, 0)`. If `p =
    /// self.max`, the offset is `(1, 1, 1)` and so on for values in between.
    #[inline]
    pub fn offset(&self, p: &TVec2<T>) -> TVec2<T> {
        let mut o = p - self.min;
        if self.max.x > self.min.x {
            o.x /= self.max.x - self.min.x
        };
        if self.max.y > self.min.y {
            o.y /= self.max.y - self.min.y
        };
        o
    }

    /// Computes the bounding sphere that encompasses this bounding box.
    #[inline]
    pub fn bounding_sphere(&self) -> (TVec2<T>, T) {
        let center = (self.min + self.max) / T::TWO;
        let radius = if self.inside(&center) { (center - self.max).magnitude() } else { T::ZERO };
        (center, radius)
    }

    /// Returns whether or not this bounding box is empty (whether it has no
    /// points in it).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y
    }

    /// Returns whether or not this bounding box is degenerate.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }

    /// Computes whether or not the given point `p` is inside the bounding box.
    #[inline]
    pub fn inside(&self, p: &TVec2<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Same as `inside`, but excludes points on the upper boundary.
    #[inline]
    pub fn inside_exclusive(&self, p: &TVec2<T>) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }

    /// Computes whether or not two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, rhs: &Self) -> bool {
        ((self.max.x >= rhs.min.x) && (self.min.x <= rhs.max.x)) && ((self.max.y >= rhs.min.y) && (self.min.y <= rhs.max.y))
    }

    /// Computes the squared distance from point `p` to the edge of the bounding
    /// box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance_sqr(&self, p: &TVec2<T>) -> T {
        let dx = T::ZERO.maxi(self.min.x - p.x).maxi(p.x - self.max.x);
        let dy = T::ZERO.maxi(self.min.y - p.y).maxi(p.y - self.max.y);
        dx * dx + dy * dy
    }

    /// Computes the distance from point `p` to the edge of the bounding box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance(&self, p: &TVec2<T>) -> T {
        self.distance_sqr(p).sqrt()
    }

    /// Pads the bounding box by a constant amount `delta` equally in all
    /// dimensions.
    #[inline]
    pub fn expand(self, delta: T) -> Self {
        Self { min: self.min - TVec2::new(delta, delta), max: self.max + TVec2::new(delta, delta) }
    }
}


impl<T: Numeric + Clone + Copy> TBounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[doc(alias = "|, |=")]
    #[inline]
    pub fn union_box(self, rhs: &Self) -> Self {
        Self {
            min: TVec2::new(self.min.x.mini(rhs.min.x), self.min.y.mini(rhs.min.y)),
            max: TVec2::new(self.max.x.maxi(rhs.max.x), self.max.y.maxi(rhs.max.y)),
        }
    }

    fn union_box_assign(&mut self, rhs: &Self) {
        self.min = TVec2::new(self.min.x.mini(rhs.min.x), self.min.y.mini(rhs.min.y));
        self.max = TVec2::new(self.max.x.maxi(rhs.max.x), self.max.y.maxi(rhs.max.y));
    }

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[doc(alias = "|, |=")]
    #[inline]
    pub fn union_vect(self, rhs: &TVec2<T>) -> Self {
        Self {
            min: TVec2::new(self.min.x.mini(rhs.x), self.min.y.mini(rhs.y)),
            max: TVec2::new(self.max.x.maxi(rhs.x), self.max.y.maxi(rhs.y)),
        }
    }

    fn union_vect_assign(&mut self, rhs: &TVec2<T>) {
        self.min = TVec2::new(self.min.x.mini(rhs.x), self.min.y.mini(rhs.y));
        self.max = TVec2::new(self.max.x.maxi(rhs.x), self.max.y.maxi(rhs.y));
    }

    /// Takes the intersection of two bounding boxes
    #[doc(alias = "&, &=")]
    #[inline]
    pub fn intersect(self, rhs: &Self) -> Self {
        Self {
            min: TVec2::new(self.min.x.maxi(rhs.min.x), self.min.y.maxi(rhs.min.y)),
            max: TVec2::new(self.max.x.mini(rhs.max.x), self.max.y.mini(rhs.max.y)),
        }
    }

    fn intersect_assign(&mut self, rhs: &Self) {
        self.min = TVec2::new(self.min.x.maxi(rhs.min.x), self.min.y.maxi(rhs.min.y));
        self.max = TVec2::new(self.max.x.mini(rhs.max.x), self.max.y.mini(rhs.max.y));
    }
}


impl<T> Index<u8> for TBounds2<T> {
    type Output = TVec2<T>;

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


impl<T: Numeric + Clone + Copy> BitOr<&TBounds2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor(self, rhs: &TBounds2<T>) -> Self::Output {
        self.union_box(rhs)
    }
}


impl<T: Numeric + Clone + Copy> BitOr<TBounds2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor(self, rhs: TBounds2<T>) -> Self::Output {
        self.union_box(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOr<TVec2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor(self, rhs: TVec2<T>) -> Self::Output {
        self.union_vect(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOr<&TVec2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor(self, rhs: &TVec2<T>) -> Self::Output {
        self.union_vect(rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitAnd<&TBounds2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand(self, rhs: &TBounds2<T>) -> Self::Output {
        self.intersect(rhs)
    }
}


impl<T: Numeric + Clone + Copy> BitAnd<TBounds2<T>> for TBounds2<T> {
    type Output = TBounds2<T>;

    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand(self, rhs: TBounds2<T>) -> Self::Output {
        self.intersect(&rhs)
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<TBounds2<T>> for TBounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: TBounds2<T>) {
        self.union_box_assign(&rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<&TBounds2<T>> for TBounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: &TBounds2<T>) {
        self.union_box_assign(rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<TVec2<T>> for TBounds2<T> {
    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: TVec2<T>) {
        self.union_vect_assign(&rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitOrAssign<&TVec2<T>> for TBounds2<T> {
    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn bitor_assign(&mut self, rhs: &TVec2<T>) {
        self.union_vect_assign(rhs);
    }
}

impl<T: Numeric + Clone + Copy> BitAndAssign<&TBounds2<T>> for TBounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand_assign(&mut self, rhs: &TBounds2<T>) {
        self.intersect_assign(rhs);
    }
}


impl<T: Numeric + Clone + Copy> BitAndAssign<TBounds2<T>> for TBounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn bitand_assign(&mut self, rhs: TBounds2<T>) {
        self.intersect_assign(&rhs);
    }
}
