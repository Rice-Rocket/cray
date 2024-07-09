use crate::{Bounds2i, Point2i};

#[derive(Debug, Clone)]
pub struct Vec2D<T>
where
    T: Default + Clone + Copy
{
    pub data: Vec<T>,
    extent: Bounds2i
}

impl<T> Vec2D<T>
where 
    T: Default + Clone + Copy
{
    pub fn from_bounds(bounds: Bounds2i) -> Self {
        let n = bounds.area();
        let data = vec![T::default(); n.try_into().unwrap()];
        Self { data, extent: bounds }
    }

    pub fn get(&self, p: Point2i) -> T {
        let (x, y) = self.xy(p);
        let width = self.width();
        self.data[(y * width + x) as usize]
    }

    pub fn get_xy(&self, x: i32, y: i32) -> T {
        let width = self.width();
        self.data[(y * width + x) as usize]
    }

    pub fn get_mut(&mut self, p: Point2i) -> &mut T {
        let (x, y) = self.xy(p);
        let width = self.width();
        &mut self.data[(y * width + x) as usize]
    }

    pub fn set(&mut self, p: Point2i, val: T) {
        let (x, y) = self.xy(p);
        let width = self.width();
        self.data[(y * width + x) as usize] = val;
    }

    pub fn width(&self) -> i32 {
        self.extent.max.x - self.extent.min.x
    }

    pub fn height(&self) -> i32 {
        self.extent.max.y - self.extent.min.y
    }

    fn xy(&self, p: Point2i) -> (i32, i32) {
        let x = p.x - self.extent.min.x;
        let y = p.y - self.extent.min.y;
        (x, y)
    }

    pub fn extent(&self) -> Bounds2i {
        self.extent
    }
}
