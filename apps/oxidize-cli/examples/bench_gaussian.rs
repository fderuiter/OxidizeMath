//! Example target for bench gaussian.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]
use oxidizemath::ai::gaussian_splatting::rendering::blend_gaussians;
use oxidizemath::ai::gaussian_splatting::structs::Gaussian2D;
use oxidizemath::nalgebra::{Matrix2, Point2, Vector3};
use std::time::Instant;

fn main() {
    let smoke = std::env::var("MATH_EXPLORER_SMOKE")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let (num_gaussians, width, height, frames) = if smoke {
        (16, 16, 16, 2)
    } else {
        (100, 100, 100, 100)
    };
    let mut gaussians = Vec::new();

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

    let start = Instant::now();

    let mut black_hole = Vector3::zeros();

    for _ in 0..frames {
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
