//! Profile Isosurface Demo
//!
//! This example benchmarks the isosurface extraction process.
#![cfg_attr(not(target_arch = "wasm32"), allow(clippy::disallowed_methods))]

use math_explorer::applied::isosurface::{Point3D, VoxelGrid, extract_isosurface};
use std::time::Instant;

fn run_micro_benchmark() {
    println!("\n[Micro-benchmark] Point3D::normalize");
    let count = 50_000_000;
    let mut vectors = Vec::with_capacity(count);

    let mut seed: u64 = 0xDEADBEEF;
    for _ in 0..count {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((seed >> 16) & 0xFFFF) as f32 / 65536.0;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = ((seed >> 16) & 0xFFFF) as f32 / 65536.0;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let z = ((seed >> 16) & 0xFFFF) as f32 / 65536.0;
        vectors.push(Point3D::new(x, y, z));
    }

    let start_micro = Instant::now();
    let mut checksum = 0.0;
    for v in &vectors {
        let n = v.normalize();
        checksum += n.x + n.y + n.z;
    }
    let duration_micro = start_micro.elapsed();
    println!("Processed {} vectors in {:?}", count, duration_micro);
    println!(
        "Average time per vector: {:.4} ns",
        (duration_micro.as_nanos() as f64) / (count as f64)
    );
    println!("Checksum: {}", checksum);
}

fn run_macro_benchmark() {
    println!("\n[Macro-benchmark] extract_isosurface (Sphere SDF)");

    let size = 128;
    println!("Grid size: {}x{}x{}", size, size, size);

    let mut data = Vec::with_capacity(size * size * size);
    let center = size as f32 / 2.0;
    let radius = size as f32 / 3.0;
    let radius_sq = radius * radius;

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let dz = z as f32 - center;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                data.push(dist_sq - radius_sq);
            }
        }
    }

    let grid = VoxelGrid::builder()
        .dimensions(size, size, size)
        .data(data)
        .voxel_size(Point3D::new(1.0, 1.0, 1.0))
        .origin(Point3D::new(0.0, 0.0, 0.0))
        .build()
        .unwrap();

    let _ = extract_isosurface(&grid, 0.0);

    let start_macro = Instant::now();
    let mesh = extract_isosurface(&grid, 0.0).expect("Failed to extract surface");
    let duration_macro = start_macro.elapsed();

    println!(
        "Extracted {} triangles in {:?}",
        mesh.indices.len() / 3,
        duration_macro
    );
}

fn main() {
    println!("⏱️  Profiler Benchmark Starting...");
    run_micro_benchmark();
    run_macro_benchmark();
}
