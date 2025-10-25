use kd_tree::KdPoint;
use pasture_algorithms::normal_estimation::compute_normals;
use pasture_core::{
    containers::{BorrowedMutBufferExt, MakeBufferFromLayout, VectorBuffer},
    layout::PointType,
    nalgebra::Vector3,
};
use pasture_derive::PointType;
use rerun::Points3D;

pub struct GradientAtPoint {
    pub location: pasture_core::nalgebra::Vector3<f64>,
    pub gradient: f64,
}

#[repr(C, packed)]
#[derive(Copy, Clone, PointType, Debug, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
struct PasturePoint {
    #[pasture(BUILTIN_POSITION_3D)]
    pub position: Vector3<f64>,
    #[pasture(BUILTIN_INTENSITY)]
    pub intensity: u16,
}
impl KdPoint for PasturePoint {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 {
        let position = self.position;
        position[k]
    }
}

pub fn pcl2gradientmap(pcl: &Vec<nalgebra::Vector3<f64>>) -> Vec<GradientAtPoint> {
    let mut buffer = VectorBuffer::new_from_layout(PasturePoint::layout());
    na_to_pasture(&pcl, &mut buffer);
    let normal_map: Vec<(Vector3<f64>, f64)> =
        compute_normals::<VectorBuffer, PasturePoint>(&buffer, 100);

    pcl.iter()
        .zip(normal_map.iter())
        .map(|(position, (normal, _curvature))| {
            let normalized = normal.normalize();

            let horizontal_component =
                (normalized.x * normalized.x + normalized.y * normalized.y).sqrt();
            let gradient = if normalized.z.abs() > 1e-10 {
                horizontal_component / normalized.z.abs()
            } else {
                f64::INFINITY
            };

            GradientAtPoint {
                location: pasture_core::nalgebra::Vector3::new(position.x, position.y, position.z),
                gradient,
            }
        })
        .collect()
}

fn na_to_pasture(pcl: &Vec<nalgebra::Vector3<f64>>, buffer: &mut VectorBuffer) {
    for point in pcl {
        buffer.view_mut().push_point(PasturePoint {
            position: pasture_core::nalgebra::Vector3::new(point.x, point.y, point.z),
            intensity: 84,
        });
    }
}

pub fn grad_map_to_rerun_points3d(map: &Vec<GradientAtPoint>) -> Points3D {
    Points3D::new(
        map.iter()
            .map(|g| [g.location.x, g.location.y, g.location.z]),
    )
    .with_colors(map.iter().map(|g| gradient_to_color(g.gradient)))
    .with_labels(map.iter().map(|g| format!("{:.2}", g.gradient)))
}

fn gradient_to_color(gradient: f64) -> [u8; 3] {
    let clamped = gradient.min(3.0).max(0.0);

    if clamped < 0.3 {
        let t = clamped / 0.3;
        [0, (100.0 + 155.0 * t) as u8, (100.0 + 55.0 * t) as u8]
    } else if clamped < 0.5 {
        let t = (clamped - 0.3) / 0.2;
        [(180.0 * t) as u8, 255, (155.0 - 55.0 * t) as u8]
    } else if clamped < 0.7 {
        let t = (clamped - 0.5) / 0.2;
        [(180.0 + 75.0 * t) as u8, 255, (100.0 - 100.0 * t) as u8]
    } else if clamped < 1.0 {
        let t = (clamped - 0.7) / 0.3;
        [255, (255.0 - 90.0 * t) as u8, 0]
    } else if clamped < 1.5 {
        // Orange to red-orange (1.0-1.5)
        let t = (clamped - 1.0) / 0.5;
        [255, (165.0 - 90.0 * t) as u8, 0]
    } else if clamped < 2.0 {
        // Red-orange to red (1.5-2.0)
        let t = (clamped - 1.5) / 0.5;
        [255, (75.0 - 75.0 * t) as u8, 0]
    } else {
        // Red to dark red (2.0-3.0)
        let t = ((clamped - 2.0) / 1.0).min(1.0);
        [(255.0 - 100.0 * t) as u8, 0, 0]
    }
}

/// Return an image where each pixel is at the height of the nearest point in the gradient map
/// The function creates a 2D grid and assigns each pixel the height (z-coordinate) of the
/// closest point from the gradient map
pub fn grad_map_to_height_field(
    map: &Vec<GradientAtPoint>,
    width: usize,
    height: usize,
) -> Vec<Vec<f64>> {
    if map.is_empty() {
        return vec![vec![0.0; width]; height];
    }

    // Find bounds of the point cloud
    let min_x = map
        .iter()
        .map(|g| g.location.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = map
        .iter()
        .map(|g| g.location.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = map
        .iter()
        .map(|g| g.location.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = map
        .iter()
        .map(|g| g.location.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_range = max_x - min_x;
    let y_range = max_y - min_y;

    let mut height_field = vec![vec![f64::NAN; width]; height];
    
    // Calculate a reasonable threshold for "unknown" pixels
    // Use a fraction of the image diagonal as the max distance
    let img_diagonal = ((x_range * x_range + y_range * y_range).sqrt() / 20.0).max(0.1);
    let max_dist_sq = img_diagonal * img_diagonal;

    // For each pixel, find the nearest point and use its z value
    for row in 0..height {
        for col in 0..width {
            // Map pixel coordinates to world coordinates
            let world_x = min_x + (col as f64 / width as f64) * x_range;
            let world_y = min_y + (row as f64 / height as f64) * y_range;

            // Find nearest point
            let mut min_dist = f64::INFINITY;
            let mut nearest_z = 0.0;

            for point in map {
                let dx = point.location.x - world_x;
                let dy = point.location.y - world_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    nearest_z = point.location.z;
                }
            }

            // Only assign height if the nearest point is close enough
            if min_dist < max_dist_sq {
                height_field[row][col] = nearest_z;
            }
        }
    }

    height_field
}

/// Convert a gradient map to an RGB image where pixels are colored based on slope:
/// - Green (low slope) to Red (high slope)
/// Uses the existing gradient_to_color function for consistent color mapping
pub fn grad_map_to_image(
    map: &Vec<GradientAtPoint>,
    width: usize,
    height: usize,
) -> Vec<Vec<[u8; 3]>> {
    if map.is_empty() {
        return vec![vec![[0, 255, 0]; width]; height];
    }

    // Find bounds of the point cloud
    let min_x = map
        .iter()
        .map(|g| g.location.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = map
        .iter()
        .map(|g| g.location.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = map
        .iter()
        .map(|g| g.location.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = map
        .iter()
        .map(|g| g.location.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_range = max_x - min_x;
    let y_range = max_y - min_y;

    let mut image = vec![vec![[255u8, 255u8, 255u8]; width]; height];
    
    // Calculate a reasonable threshold for "unknown" pixels
    // Use a fraction of the image diagonal as the max distance
    let img_diagonal = ((x_range * x_range + y_range * y_range).sqrt() / 20.0).max(0.1);
    let max_dist_sq = img_diagonal * img_diagonal;

    // For each pixel, find the nearest point and use its gradient for coloring
    for row in 0..height {
        for col in 0..width {
            // Map pixel coordinates to world coordinates
            let world_x = min_x + (col as f64 / width as f64) * x_range;
            let world_y = min_y + (row as f64 / height as f64) * y_range;

            // Find nearest point
            let mut min_dist = f64::INFINITY;
            let mut nearest_gradient = 0.0;

            for point in map {
                let dx = point.location.x - world_x;
                let dy = point.location.y - world_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < min_dist {
                    min_dist = dist_sq;
                    nearest_gradient = point.gradient;
                }
            }

            // Only color the pixel if the nearest point is close enough
            // Otherwise leave it white (unknown)
            if min_dist < max_dist_sq {
                image[row][col] = gradient_to_color(nearest_gradient);
            }
        }
    }

    image
}
