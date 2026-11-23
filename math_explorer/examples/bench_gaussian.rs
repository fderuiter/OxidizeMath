#![allow(missing_docs)]
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use math_explorer::ai::gaussian_splatting::rendering::blend_gaussians;
use math_explorer::ai::gaussian_splatting::structs::Gaussian2D;
use math_explorer::nalgebra::{Matrix2, Point2, Vector3};
use std::time::Instant;

fn main() {
    let mut gaussians = Vec::new();
    let num_gaussians = 100;

    // Create gaussians
    // We use identity covariance for simplicity as it's invertible.
    for i in 0..num_gaussians {
        gaussians.push(Gaussian2D {
            mean: Point2::new((i as f64) % 10.0, (i as f64) / 10.0),
            conic: Matrix2::from_diagonal_element(-0.5),
            opacity: 0.5,
            color: Vector3::new(1.0, 0.0, 0.0),
            depth: 1.0,
        });
    }

    let width = 100;
    let height = 100;

    let start = Instant::now();

    let mut black_hole = Vector3::zeros();

    // 100 iterations of the frame
    for _ in 0..100 {
        for y in 0..height {
            for x in 0..width {
                let p = Point2::new(x as f64 / 10.0, y as f64 / 10.0);
                black_hole += blend_gaussians(&gaussians, &p);
            }
        }
    }

    let duration = start.elapsed();
    println!("Time: {:?}", duration);
    // Prevent optimization
    if black_hole.x > 1e9 {
        println!("Overflow");
    }
}
