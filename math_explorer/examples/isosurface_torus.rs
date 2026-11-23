#![allow(missing_docs)]
use math_explorer::applied::isosurface::{Point3D, VoxelGrid, extract_isosurface};
use std::fmt;
use std::fs::File;
use std::io::Write;

// Simple ANSI color wrapper for better UX
enum Color {
    Cyan,
    Green,
    Yellow,
    Magenta,
    Bold,
    Reset,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let code = match self {
            Color::Cyan => "\x1b[36m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Magenta => "\x1b[35m",
            Color::Bold => "\x1b[1m",
            Color::Reset => "\x1b[0m",
        };
        write!(f, "{}", code)
    }
}

fn generate_torus_sdf(
    width: usize,
    height: usize,
    depth: usize,
    major_radius: f32,
    minor_radius: f32,
) -> math_explorer::error::Result<VoxelGrid> {
    let min_bound = -4.0;
    let max_bound = 4.0;
    let range = max_bound - min_bound;
    let step = range / (width as f32);

    let mut data = Vec::with_capacity(width * height * depth);

    for z in 0..depth {
        let z_coord = min_bound + (z as f32) * step;
        for y in 0..height {
            let y_coord = min_bound + (y as f32) * step;
            for x in 0..width {
                let x_coord = min_bound + (x as f32) * step;
                let len_xy = (x_coord * x_coord + y_coord * y_coord).sqrt();
                let dist =
                    ((len_xy - major_radius).powi(2) + z_coord * z_coord).sqrt() - minor_radius;
                data.push(dist);
            }
        }
    }

    let grid = VoxelGrid::builder()
        .dimensions(width, height, depth)
        .data(data)
        .voxel_size(Point3D::new(step, step, step))
        .origin(Point3D::new(min_bound, min_bound, min_bound))
        .build()?;
    Ok(grid)
}

/// This example generates a mesh for a Torus using the Marching Cubes algorithm
/// and exports it to a Wavefront OBJ file.
fn main() -> math_explorer::error::Result<()> {
    // Header
    println!();
    println!("{}🍩 Torus Generator 3000{}", Color::Magenta, Color::Reset);
    println!("{}======================={}", Color::Magenta, Color::Reset);
    println!();

    println!(
        "{}⚙️  Generating Torus Signed Distance Field (SDF)...{}",
        Color::Yellow,
        Color::Reset
    );

    let width = 64;
    let height = 64;
    let depth = 64;
    let major_radius = 2.0;
    let minor_radius = 0.8;

    println!(
        "   Dimensions: {}{}x{}x{}{}",
        Color::Bold,
        width,
        height,
        depth,
        Color::Reset
    );
    println!(
        "   Radii:      Major={}{:.1}{}, Minor={}{:.1}{}",
        Color::Cyan,
        major_radius,
        Color::Reset,
        Color::Cyan,
        minor_radius,
        Color::Reset
    );
    println!();

    let grid = generate_torus_sdf(width, height, depth, major_radius, minor_radius)?;

    println!(
        "{}⛏️  Extracting Isosurface using Marching Cubes...{}",
        Color::Yellow,
        Color::Reset
    );
    let mesh = extract_isosurface(&grid, 0.0)?;
    println!(
        "   Generated {}{}{} triangles.",
        Color::Bold,
        mesh.indices.len() / 3,
        Color::Reset
    );
    println!();

    // Export to OBJ
    let filename = "torus.obj";
    let obj_str = math_explorer::mesh::export_mesh_to_obj_string(&mesh)?;
    let mut file = File::create(filename)?;
    file.write_all(obj_str.as_bytes())?;

    println!("{}✅ Success!{}", Color::Green, Color::Reset);
    println!(
        "   Mesh saved to: {}{}{}",
        Color::Cyan,
        filename,
        Color::Reset
    );
    println!("   You can view this file in Blender, MeshLab, or any 3D viewer.");
    println!();

    Ok(())
}
