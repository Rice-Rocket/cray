use crate::{Bounds2i, Point2i};

pub struct Tile {
    pub bounds: Bounds2i,
}

impl Tile {
    pub fn tiles(orig_bounds: Bounds2i, tile_width: i32, tile_height: i32) -> Vec<Tile> {
        let im_width = orig_bounds.width();
        let im_height = orig_bounds.height();
        let num_horizontal_tiles = im_width / tile_width;
        let rem_horizontal_pixels = im_width % tile_width;
        let num_vertical_tiles = im_height / tile_height;
        let rem_vertical_pixels = im_height % tile_height;

        let mut tiles = Vec::with_capacity((num_horizontal_tiles * num_vertical_tiles) as usize);

        for tile_y in 0..num_vertical_tiles {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = orig_bounds.min.x + tile_x * tile_width;
                let tile_start_y = orig_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i::new(tile_start_x, tile_start_y),
                        Point2i::new(tile_start_x + tile_width, tile_start_y + tile_height),
                    )
                });
            }

            if rem_horizontal_pixels > 0 {
                let tile_start_x = orig_bounds.min.x + num_horizontal_tiles * tile_width;
                let tile_start_y = orig_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i::new(tile_start_x, tile_start_y),
                        Point2i::new(tile_start_x + rem_horizontal_pixels, tile_start_y + tile_height),
                    )
                })
            }
        }

        if rem_vertical_pixels > 0 {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = orig_bounds.min.x + tile_x * tile_width;
                let tile_start_y = orig_bounds.min.y + num_vertical_tiles * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i::new(tile_start_x, tile_start_y),
                        Point2i::new(tile_start_x + tile_width, tile_start_y + rem_vertical_pixels)
                    )
                })
            }
        }

        if rem_horizontal_pixels > 0 && rem_vertical_pixels > 0 {
            let tile_start_x = orig_bounds.min.x + num_horizontal_tiles * tile_width;
            let tile_start_y = orig_bounds.min.y + num_vertical_tiles * tile_height;
            tiles.push(Tile {
                bounds: Bounds2i::new(
                    Point2i::new(tile_start_x, tile_start_y),
                    Point2i::new(tile_start_x + rem_horizontal_pixels, tile_start_y + rem_vertical_pixels),
                )
            });
        }

        tiles
    }
}
