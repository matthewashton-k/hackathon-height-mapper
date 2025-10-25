use rerun::{Points3D, RecordingStreamBuilder, Transform3D, ViewCoordinates};
use simple_icp::point3d::Point3d;
use std::time::{Duration, Instant};

use crate::{
    convert_to_slopes::{grad_map_to_rerun_points3d, pcl2gradientmap},
    icp::Pipeline,
};
pub mod convert_to_slopes;
pub mod icp;

fn main() {
    let mut pipeline = Pipeline::new();
    let mut last_grad_map_log = Instant::now();
    let grad_map_log_interval = Duration::from_secs(5);
    let ip = "192.168.0.108";
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
            last_grad_map_log = Instant::now();
        }
    }
}
