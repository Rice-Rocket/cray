use crate::{color::sampled::SampledSpectrum, ray::AbstractRay, Bounds3f, Float, Point3f, Point3i, Ray, Vec3f};

use super::RayMajorantSegment;

pub enum RayMajorantIterator<'a> {
    Homogeneous(HomogeneousMajorantIterator),
    DDA(DDAMajorantIterator<'a>),
}

impl Iterator for RayMajorantIterator<'_> {
    type Item = RayMajorantSegment;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RayMajorantIterator::Homogeneous(m) => m.next(),
            RayMajorantIterator::DDA(m) => m.next(),
        }
    }
}

pub struct HomogeneousMajorantIterator {
    seg: RayMajorantSegment,
    called: bool,
}

impl HomogeneousMajorantIterator {
    pub fn new(t_min: Float, t_max: Float, sigma_maj: SampledSpectrum) -> HomogeneousMajorantIterator {
        HomogeneousMajorantIterator {
            seg: RayMajorantSegment { t_min, t_max, sigma_maj },
            called: false,
        }
    }
}

impl Iterator for HomogeneousMajorantIterator {
    type Item = RayMajorantSegment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.called { return None };
        self.called = true;
        Some(self.seg)
    }
}

pub struct DDAMajorantIterator<'a> {
    sigma_t: SampledSpectrum,
    t_min: Float,
    t_max: Float,
    grid: &'a MajorantGrid,
    next_crossing_t: [Float; 3],
    delta_t: [Float; 3],
    step: [i32; 3],
    voxel_limit: [i32; 3],
    voxel: [i32; 3],
}

impl<'a> DDAMajorantIterator<'a> {
    pub fn new(
        ray: Ray,
        t_min: Float,
        t_max: Float,
        grid: &'a MajorantGrid,
        sigma_t: SampledSpectrum,
    ) -> DDAMajorantIterator {
        let diag = grid.bounds.diagonal();
        let mut ray_grid = Ray::new(
            grid.bounds.offset(ray.origin),
            Vec3f::from(ray.direction / diag),
        );
        let grid_intersect = ray_grid.at(t_min);

        let mut next_crossing_t = [0.0; 3];
        let mut delta_t = [0.0; 3];
        let mut step = [0; 3];
        let mut voxel_limit = [0; 3];
        let mut voxel = [0; 3];

        for axis in 0..3 {
            voxel[axis] = i32::clamp((grid_intersect[axis] * grid.res[axis] as Float) as i32, 0, grid.res[axis] - 1);
            delta_t[axis] = 1.0 / (ray_grid.direction[axis].abs() * grid.res[axis] as Float);
            if ray_grid.direction[axis] == -0.0 {
                ray_grid.direction[axis] = 0.0;
            }

            if ray_grid.direction[axis] >= 0.0 {
                let next_voxel_pos = (voxel[axis] + 1) as Float / grid.res[axis] as Float;
                next_crossing_t[axis] = t_min + (next_voxel_pos - grid_intersect[axis]) / ray_grid.direction[axis];
                step[axis] = 1;
                voxel_limit[axis] = grid.res[axis];
            } else {
                let next_voxel_pos = voxel[axis] as Float / grid.res[axis] as Float;
                next_crossing_t[axis] = t_min + (next_voxel_pos - grid_intersect[axis]) / ray_grid.direction[axis];
                step[axis] = -1;
                voxel_limit[axis] = -1;
            }
        }

        DDAMajorantIterator {
            sigma_t,
            t_min,
            t_max,
            grid,
            next_crossing_t,
            delta_t,
            step,
            voxel_limit,
            voxel,
        }
    }
}

impl Iterator for DDAMajorantIterator<'_> {
    type Item = RayMajorantSegment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.t_min >= self.t_max { return None };

        let bits = if self.next_crossing_t[0] < self.next_crossing_t[1] { 4 } else { 0 }
            | if self.next_crossing_t[0] < self.next_crossing_t[2] { 2 } else { 0 }
            | if self.next_crossing_t[1] < self.next_crossing_t[2] { 1 } else { 0 };

        const CMP_TO_AXIS: [usize; 8] = [2, 1, 2, 1, 2, 2, 0, 0];
        let step_axis = CMP_TO_AXIS[bits];
        let t_voxel_exit = self.t_max.min(self.next_crossing_t[step_axis]);

        let sigma_maj = self.sigma_t * self.grid.lookup(self.voxel[0], self.voxel[1], self.voxel[2]);
        let seg = RayMajorantSegment { t_min: self.t_min, t_max: t_voxel_exit, sigma_maj };

        self.t_min = t_voxel_exit;
        if self.next_crossing_t[step_axis] > self.t_max {
            self.t_min = self.t_max;
        }

        self.voxel[step_axis] += self.step[step_axis];
        if self.voxel[step_axis] == self.voxel_limit[step_axis] {
            self.t_min = self.t_max;
        }

        self.next_crossing_t[step_axis] += self.delta_t[step_axis];

        Some(seg)
    }
}

#[derive(Debug, Clone)]
pub struct MajorantGrid {
    pub bounds: Bounds3f,
    pub voxels: Vec<Float>,
    pub res: Point3i,
}

impl MajorantGrid {
    pub fn new(bounds: Bounds3f, res: Point3i) -> MajorantGrid {
        MajorantGrid {
            bounds,
            res,
            voxels: vec![0.0; (res.x * res.y * res.z) as usize],
        }
    }

    pub fn lookup(&self, x: i32, y: i32, z: i32) -> Float {
        self.voxels[(x + self.res.x * (y + self.res.y * z)) as usize]
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, v: Float) {
        self.voxels[(x + self.res.x * (y + self.res.y * z)) as usize] = v;
    }

    pub fn voxel_bounds(&self, x: i32, y: i32, z: i32) -> Bounds3f {
        let p0 = Point3f::new(
            x as Float / self.res.x as Float,
            y as Float / self.res.y as Float,
            z as Float / self.res.z as Float,
        );
        let p1 = Point3f::new(
            (x + 1) as Float / self.res.x as Float,
            (y + 1) as Float / self.res.y as Float,
            (z + 1) as Float / self.res.z as Float,
        );

        Bounds3f::new(p0, p1)
    }
}
