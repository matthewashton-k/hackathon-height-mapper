use std::{
    collections::HashSet,
    sync::mpsc::{Receiver, channel},
    thread::{self, JoinHandle},
    time::Duration,
};

use realsense_rust::{
    context::Context,
    frame::{DepthFrame, PixelKind},
    kind::{Rs2Format, Rs2ProductLine, Rs2StreamKind},
    pipeline::{FrameWaitError, InactivePipeline},
};
use simple_icp::{config::Config, icp_pipeline::IcpPipeline};

pub struct DepthInfo {
    focal_length_px: f32,
    frame_width: u32,
    frame_height: u32,
}

pub struct Pipeline {
    pub icp: IcpPipeline,
    pub realsense_publisher: Receiver<Vec<Vector3<f32>>>,
    pub realsense_publish_thread: JoinHandle<()>,
}

impl Pipeline {
    pub fn new() -> Self {
        let mut icp_config = Config::default_values();
        icp_config.deskew = false;
        icp_config.max_points_per_voxel = 200;
        icp_config.voxel_size = 0.2;
        let icp = IcpPipeline::new_with_config(icp_config);
        let (handle, rx) = Self::create_realsense_publisher().unwrap();
        Self {
            icp,
            realsense_publisher: rx,
            realsense_publish_thread: handle,
        }
    }

    fn create_realsense_publisher()
    -> Result<(JoinHandle<()>, Receiver<Vec<Vector3<f32>>>), Box<dyn std::error::Error>> {
        let (tx, rx) = channel();
        let handle = thread::spawn(move || {
            let mut queried_devices = HashSet::new();
            queried_devices.insert(Rs2ProductLine::D400);
            let context = Context::new().unwrap();
            let devices = context.query_devices(queried_devices);
            let pipeline = InactivePipeline::try_from(&context).unwrap();
            let mut config = realsense_rust::config::Config::new();
            config.disable_all_streams().unwrap();
            config
                .enable_stream(Rs2StreamKind::Depth, None, 640, 480, Rs2Format::Z16, 30)
                .unwrap();
            let mut pipeline = pipeline.start(Some(config)).unwrap();
            let mut stream = None;
            for s in pipeline.profile().streams() {
                let _ = match s.format() {
                    Rs2Format::Z16 => stream = Some(s),
                    _format => {
                        continue;
                    }
                };
            }
            let depth_format = stream.unwrap().intrinsics().unwrap();
            let focal_length_px;
            if depth_format.fx() != depth_format.fy() {
                focal_length_px = (depth_format.fx() + depth_format.fy()) / 2.0;
            } else {
                focal_length_px = depth_format.fx();
            }
            println!("focal_length_px {focal_length_px}");
            loop {
                let frames = loop {
                    match pipeline.wait(Some(Duration::from_millis(2000))) {
                        Ok(x) => {
                            break x;
                        }
                        Err(e) => {
                            eprintln!("Failed to get frame from RealSense Camera {e}",);
                            continue;
                        }
                    }
                };
                for frame in frames.frames_of_type::<DepthFrame>() {
                    let depth_scale = frame.depth_units().unwrap();
                    if !matches!(frame.get(0, 0), Some(PixelKind::Z16 { .. })) {
                        eprintln!("Unexpected depth pixel kind for camera");
                    }
                    debug_assert_eq!(frame.bits_per_pixel(), 16);
                    debug_assert_eq!(frame.width() * frame.height() * 2, frame.get_data_size());

                    let slice;
                    unsafe {
                        let data: *const _ = frame.get_data();
                        slice = std::slice::from_raw_parts(
                            data.cast::<u16>(),
                            frame.width() * frame.height(),
                        );
                    }
                    let Ok(depth_scale) = frame.depth_units() else {
                        continue;
                    };
                    let frame = slice.to_vec();
                    let pcl = depth_to_point_cloud(&frame, depth_scale, 3.0, 1);
                    tx.send(pcl).unwrap();
                }
            }
        });
        return Ok((handle, rx));
    }
}

use nalgebra::Vector3;

pub fn depth_to_point_cloud(
    depths: &[u16],
    depth_scale: f32,
    max_depth: f32,
    stride: u32,
) -> Vec<Vector3<f32>> {
    const PPX: f32 = 317.44882;
    const PPY: f32 = 245.91605;
    const FOCAL_LENGTH_PX: f32 = 394.40106;
    const FRAME_WIDTH: u32 = 640;
    const FRAME_HEIGHT: u32 = 480;

    let stride = stride.max(1);
    let width_over_stride = FRAME_WIDTH / stride;
    let height_over_stride = FRAME_HEIGHT / stride;

    let mut points = Vec::new();

    for strided_y in 0..height_over_stride {
        for strided_x in 0..width_over_stride {
            let x_pixel = strided_x * stride;
            let y_pixel = strided_y * stride;

            if x_pixel >= FRAME_WIDTH {
                continue;
            }

            let original_i = (x_pixel + y_pixel * FRAME_WIDTH) as usize;

            if original_i >= depths.len() {
                continue;
            }

            let depth_u = depths[original_i];

            // Skip invalid depth
            if depth_u == 0 {
                continue;
            }

            let depth = depth_u as f32 * depth_scale;

            // Skip points beyond max depth
            if depth > max_depth {
                continue;
            }

            // Convert pixel coordinates to camera coordinates
            let x = x_pixel as f32 - PPX;
            let y = y_pixel as f32 - PPY;

            let new_scale = depth / FOCAL_LENGTH_PX;

            let point_x = x * new_scale;
            let point_y = -y * new_scale;
            let point_z = -depth;

            let intermediate = Vector3::new(-point_z, -point_x, point_y);

            let transformed_point = Vector3::new(
                intermediate.z,  // new x = old z
                intermediate.y,  // new y = old y
                -intermediate.x, // new z = -old x
            );
            points.push(transformed_point);
        }
    }

    points
}
