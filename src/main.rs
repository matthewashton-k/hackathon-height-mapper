use rerun::{RecordingStreamBuilder, Transform3D, ViewCoordinates};
use simple_icp::point3d::Point3d;
use std::time::{Duration, Instant};

use crate::{
    convert_to_slopes::{
        grad_map_to_height_field, grad_map_to_image, grad_map_to_rerun_points3d, pcl2gradientmap,
    },
    icp::Pipeline,
};
pub mod convert_to_slopes;
pub mod icp;

fn main() {
    let mut pipeline = Pipeline::new();
    let mut last_grad_map_log = Instant::now();
    let grad_map_log_interval = Duration::from_secs(5);
    let ip = "172.20.10.3";
    let recording_stream = RecordingStreamBuilder::new("depth_mapper")
        .connect_grpc_opts(&format!("rerun+http://{ip}:9876/proxy"))
        .unwrap();
    recording_stream
        .log_static("/", &ViewCoordinates::RIGHT_HAND_Z_UP())
        .unwrap();
    recording_stream
        .log_static(
            format!("bot/xyz"),
            &rerun::Arrows3D::from_vectors([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .with_colors([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
                .with_labels(vec!["x", "y", "z"]),
        )
        .unwrap();
    while let Ok(point_cloud) = pipeline.realsense_publisher.recv() {
        let icp_frame = point_cloud
            .iter()
            .map(|p| Point3d {
                x: p.x,
                y: p.y,
                z: p.z,
                intensity: 100.0,
            })
            .collect::<Vec<Point3d>>();
        pipeline.icp.process_frame(&icp_frame);
        // recording_stream
        //     .log(
        //         "map",
        //         &Points3D::new(
        //             pipeline
        //                 .icp
        //                 .get_last_batch_points()
        //                 .iter()
        //                 .map(|p| [p.x, p.y, p.z]),
        //         ),
        //     )
        //     .unwrap();
        let isometry = pipeline.icp.t_origin_current;
        recording_stream
            .log(
                "bot/xyz",
                &Transform3D::from_translation_rotation(
                    isometry.translation.vector.cast::<f32>().data.0[0],
                    rerun::Quaternion::from_xyzw(
                        isometry.rotation.as_vector().cast::<f32>().data.0[0],
                    ),
                ),
            )
            .unwrap();

        // Only log gradient map every 5 seconds
        if last_grad_map_log.elapsed() >= grad_map_log_interval {
            let grad_map = pcl2gradientmap(&pipeline.icp.get_global_map());
            let visualizable = grad_map_to_rerun_points3d(&grad_map);
            recording_stream.log("grad_map", &visualizable).unwrap();

            // Generate and save gradient map image
            let img_width = 800;
            let img_height = 800;
            let grad_map_img = grad_map_to_image(&grad_map, img_width, img_height);

            // Convert to image format and save
            let mut img_buffer = image::RgbImage::new(img_width as u32, img_height as u32);
            for (y, row) in grad_map_img.iter().enumerate() {
                for (x, pixel) in row.iter().enumerate() {
                    img_buffer.put_pixel(x as u32, y as u32, image::Rgb(*pixel));
                }
            }
            img_buffer.save("gradient_map.png").unwrap();

            // Log gradient map image to rerun
            let img_data: Vec<u8> = grad_map_img
                .iter()
                .flat_map(|row| row.iter().flat_map(|rgb| rgb.iter().copied()))
                .collect();
            recording_stream
                .log(
                    "gradient_map_image",
                    &rerun::Image::new(
                        img_data,
                        rerun::ImageFormat::rgb8([img_width as u32, img_height as u32]),
                    ),
                )
                .unwrap();

            // Generate and save height map
            let height_map = grad_map_to_height_field(&grad_map, img_width, img_height);

            // Normalize height map to 0-255 range for visualization (ignoring NaN values)
            let min_height = height_map
                .iter()
                .flat_map(|row| row.iter())
                .cloned()
                .filter(|h| !h.is_nan())
                .fold(f64::INFINITY, f64::min);
            let max_height = height_map
                .iter()
                .flat_map(|row| row.iter())
                .cloned()
                .filter(|h| !h.is_nan())
                .fold(f64::NEG_INFINITY, f64::max);
            let height_range = max_height - min_height;

            let mut height_img = image::GrayImage::new(img_width as u32, img_height as u32);
            for (y, row) in height_map.iter().enumerate() {
                for (x, &height) in row.iter().enumerate() {
                    let pixel_value = if height.is_nan() {
                        255 // White for unknown areas
                    } else if height_range > 0.0 {
                        ((height - min_height) / height_range * 255.0) as u8
                    } else {
                        128
                    };
                    height_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
                }
            }
            height_img.save("height_map.png").unwrap();

            // Log height map to rerun as depth image
            let depth_data: Vec<f32> = height_map
                .iter()
                .flat_map(|row| row.iter().map(|&h| if h.is_nan() { f32::NAN } else { h as f32 }))
                .collect();
            recording_stream
                .log(
                    "height_map_image",
                    &rerun::DepthImage::try_from(rerun::TensorData::new(
                        vec![img_height as u64, img_width as u64],
                        rerun::TensorBuffer::F32(depth_data.into()),
                    ))
                    .unwrap(),
                )
                .unwrap();

            last_grad_map_log = Instant::now();
        }
    }
}
