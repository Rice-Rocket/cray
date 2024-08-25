// Pbrt 3.7 Bounding Boxes

use std::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, Index, Mul, Sub};

use bounds::Union;

use crate::math::*;

/// A 2-dimensional axis-aligned bounding box of type `T`
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Bounds2<T: Clone + Copy> {
    pub min: Point2<T>,
    pub max: Point2<T>,
}

impl<T> Bounds2<T>
where
    T: NumericConsts + NumericOrd + PartialOrd + Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>
{
    /// Creates a new [`TBounds2`] with the given `min` and `max`.
    #[inline]
    pub fn new(p1: Point2<T>, p2: Point2<T>) -> Self {
        Self { min: p1.min(p2), max: p1.max(p2) }
    }

    /// Creates a new [`TBounds2`] containing a single `point`.
    #[inline]
    pub const fn from_point(point: Point2<T>) -> Self {
        Self { min: point, max: point }
    }

    /// Creates a new [`TBounds2`] containing the given `points`.
    #[inline]
    pub fn from_points(points: Vec<Point2<T>>) -> Self {
        points.iter().fold(Bounds2::default(), |bounds, p| bounds.union_vect(*p))
    }

    /// Returns the position of the given `corner`.
    #[inline]
    pub fn corner(self, corner: u8) -> Point2<T> {
        Point2::new(self[corner & 1].x, self[if corner & 2 != 0 { 1 } else { 0 }].y)
    }

    /// Returns the diagonal vector of the bounding box.
    ///
    /// In other words, the vector that points from `self.min` to `self.max`.
    #[inline]
    #[doc(alias = "dimensions")]
    pub fn diagonal(self) -> Point2<T> {
        self.max - self.min
    }

    /// Returns the dimensions of the bounding box.
    #[inline(always)]
    #[doc(alias = "diagonal")]
    pub fn dimensions(self) -> Point2<T> {
        self.diagonal()
    }

    /// Computes the area of the bounding box.
    #[inline]
    pub fn area(self) -> T {
        let d = self.diagonal();
        d.x * d.y
    }

    /// Computes the longest dimension of the bounding box.
    #[inline]
    pub fn max_dim(self) -> Dimension {
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
    pub fn lerp(self, t: Point2<T>) -> Point2<T> {
        Point2::new(lerp(self.min.x, self.max.x, t.x), lerp(self.min.y, self.max.y, t.y))
    }

    /// Computes the position of a point `p` relative to the corners of the box.
    ///
    /// That is, if `p = self.min`, the offset is `(0, 0, 0)`. If `p =
    /// self.max`, the offset is `(1, 1, 1)` and so on for values in between.
    #[inline]
    pub fn offset(self, p: Point2<T>) -> Point2<T> {
        let mut o = p - self.min;
        if self.max.x > self.min.x {
            o.x = o.x / (self.max.x - self.min.x)
        };
        if self.max.y > self.min.y {
            o.y = o.y / (self.max.y - self.min.y)
        };
        o
    }

    /// Returns whether or not this bounding box is empty (whether it has no
    /// points in it).
    #[inline]
    pub fn is_empty(self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y
    }

    /// Returns whether or not this bounding box is degenerate.
    #[inline]
    pub fn is_degenerate(self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }

    /// Computes whether or not the given point `p` is inside the bounding box.
    #[inline]
    pub fn inside(self, p: Point2<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Same as `inside`, but excludes points on the upper boundary.
    #[inline]
    pub fn inside_exclusive(self, p: Point2<T>) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }

    /// Computes whether or not two bounding boxes overlap.
    #[inline]
    pub fn overlaps(self, rhs: Self) -> bool {
        ((self.max.x >= rhs.min.x) && (self.min.x <= rhs.max.x)) && ((self.max.y >= rhs.min.y) && (self.min.y <= rhs.max.y))
    }

    /// Computes the squared distance from point `p` to the edge of the bounding
    /// box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance_sqr(self, p: Point2<T>) -> T {
        let dx = T::ZERO.nmax(self.min.x - p.x).nmax(p.x - self.max.x);
        let dy = T::ZERO.nmax(self.min.y - p.y).nmax(p.y - self.max.y);
        dx * dx + dy * dy
    }

    /// Pads the bounding box by a constant amount `delta` equally in all
    /// dimensions.
    #[inline]
    pub fn expand(self, delta: T) -> Self {
        Self { min: self.min - Point2::new(delta, delta), max: self.max + Point2::new(delta, delta) }
    }

    pub fn width(&self) -> T {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> T {
        self.max.y - self.min.y
    }
}

impl<T> Default for Bounds2<T>
where
    T: NumericConsts + NumericOrd + PartialOrd + Clone + Copy + Add<T, Output = T> + Mul<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>
{
    fn default() -> Self {
        Self {
            min: Point2::MAX,
            max: Point2::MIN,
        }
    }
}


impl<T> Bounds2<T> 
where 
    T: NumericField + NumericOrd + NumericNegative + NumericFloat + Mul<Float, Output = T>
{
    /// Computes the distance from point `p` to the edge of the bounding box.
    ///
    /// If `p` is inside the bounding box, the returned distance is zero.
    #[inline]
    pub fn distance(self, p: Point2<T>) -> T {
        self.distance_sqr(p).nsqrt()
    }

    /// Computes the bounding sphere that encompasses this bounding box.
    #[inline]
    pub fn bounding_sphere(self) -> (Point2<T>, T) {
        let center = (self.min + self.max) / T::TWO;
        let radius = if self.inside(center) { (center - self.max).length() } else { T::ZERO };
        (center, radius)
    }
}


impl<T: NumericOrd + Clone + Copy> Bounds2<T> {
    /// Takes the union of two bounding boxes, extending the `min` and
    /// `max` as needed.
    #[inline]
    fn union_box(self, rhs: Self) -> Self {
        Self {
            min: Point2::new(self.min.x.nmin(rhs.min.x), self.min.y.nmin(rhs.min.y)),
            max: Point2::new(self.max.x.nmax(rhs.max.x), self.max.y.nmax(rhs.max.y)),
        }
    }

    /// Takes the union of a vector with this bounding box, extending the `min`
    /// and `max` of the bounding box as needed.
    #[inline]
    fn union_vect(self, rhs: Point2<T>) -> Self {
        Self {
            min: Point2::new(self.min.x.nmin(rhs.x), self.min.y.nmin(rhs.y)),
            max: Point2::new(self.max.x.nmax(rhs.x), self.max.y.nmax(rhs.y)),
        }
    }

    /// Takes the intersection of two bounding boxes
    #[inline]
    pub fn intersect(self, rhs: Self) -> Self {
        Self {
            min: Point2::new(self.min.x.nmax(rhs.min.x), self.min.y.nmax(rhs.min.y)),
            max: Point2::new(self.max.x.nmin(rhs.max.x), self.max.y.nmin(rhs.max.y)),
        }
    }
}

impl<T: Clone + Copy> Index<u8> for Bounds2<T> {
    type Output = Point2<T>;

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

impl<T: NumericOrd + Clone + Copy> Union<Bounds2<T>> for Bounds2<T> {
    type Output = Bounds2<T>;

    fn union(self, rhs: Bounds2<T>) -> Self::Output {
        self.union_box(rhs)
    }
}

impl<T: NumericOrd + Clone + Copy> Union<Point2<T>> for Bounds2<T> {
    type Output = Bounds2<T>;

    fn union(self, rhs: Point2<T>) -> Self::Output {
        self.union_vect(rhs)
    }
}
