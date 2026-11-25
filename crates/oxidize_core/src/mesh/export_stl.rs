//! Binary and ASCII stereolithography (`.stl`) serializers and round-trip parser.

use super::export::{invalid_mesh_error, resolve_triangle_normal, validate_mesh_geometry};
use super::types::{Mesh, Point3D};
use std::fmt::Write as _;
use std::io::{Error, ErrorKind};

/// Binary STL 80-byte header prefix declaring millimeter (`UNITS=mm`) coordinates for slicers.
pub const STL_HEADER_TEXT: &[u8] = b"OxidizeMath binary STL export; UNITS=mm";
/// Binary STL 80-byte header prefix for uncalibrated dimensionless meshes (omitting `UNITS=mm`).
pub const STL_UNITLESS_HEADER_TEXT: &[u8] = b"OxidizeMath binary STL export";

/// Minimum supported physical print span in millimeters.
pub const MIN_EXPORT_SPAN_MM: f32 = 5.0;
/// Maximum supported physical print span in millimeters.
pub const MAX_EXPORT_SPAN_MM: f32 = 250.0;
/// Default physical print span in millimeters (`100.0 mm`) when auto-calibrating unscaled meshes.
pub const DEFAULT_EXPORT_SPAN_MM: f32 = 100.0;

/// Calibrated 3D triangle mesh guaranteed to have finite vertices scaled to `[5.0, 250.0] mm`.
#[derive(Debug, Clone)]
pub struct MillimeterMesh {
    mesh: Mesh,
    span_mm: math_scalars::Millimeters<f32>,
}

impl MillimeterMesh {
    /// Centers `mesh` at the origin and scales its largest bounding-box axis span to `target_span_mm` (`[5.0, 250.0] mm`).
    pub fn from_scaled(mut mesh: Mesh, target_span_mm: f32) -> Result<Self, Error> {
        scale_mesh_to_millimeters(&mut mesh, target_span_mm)?;
        Ok(Self {
            mesh,
            span_mm: math_scalars::Millimeters::new(target_span_mm),
        })
    }

    /// Centers `mesh` at the origin and scales its largest axis span to a typed [`math_scalars::Millimeters`] quantity.
    pub fn from_scaled_quantity(
        mesh: Mesh,
        target_span: math_scalars::Millimeters<f32>,
    ) -> Result<Self, Error> {
        Self::from_scaled(mesh, target_span.get())
    }

    /// Borrows the calibrated millimeter [`Mesh`].
    #[must_use]
    pub const fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Consumes the wrapper and returns the calibrated millimeter [`Mesh`].
    #[must_use]
    pub fn into_mesh(self) -> Mesh {
        self.mesh
    }

    /// Returns the calibrated bounding-box span in typed [`math_scalars::Millimeters`].
    #[must_use]
    pub const fn span_mm(&self) -> math_scalars::Millimeters<f32> {
        self.span_mm
    }
}

/// Opaque seal token preventing external implementations of [`StlBinarySource`].
#[doc(hidden)]
pub struct StlBinarySeal(());

/// Sealed source trait distinguishing calibrated [`MillimeterMesh`] (`UNITS=mm`) from unscaled [`Mesh`] (unitless header).
pub trait StlBinarySource {
    /// Returns the underlying triangle [`Mesh`] and the matching 80-byte binary STL header prefix.
    #[doc(hidden)]
    fn stl_mesh_and_header(&self, _seal: StlBinarySeal) -> (&Mesh, &'static [u8]);
}

impl StlBinarySource for MillimeterMesh {
    fn stl_mesh_and_header(&self, _seal: StlBinarySeal) -> (&Mesh, &'static [u8]) {
        (&self.mesh, STL_HEADER_TEXT)
    }
}

impl StlBinarySource for Mesh {
    fn stl_mesh_and_header(&self, _seal: StlBinarySeal) -> (&Mesh, &'static [u8]) {
        (self, STL_UNITLESS_HEADER_TEXT)
    }
}

fn write_point_le(buf: &mut Vec<u8>, pt: Point3D) {
    buf.extend_from_slice(&pt.x.to_le_bytes());
    buf.extend_from_slice(&pt.y.to_le_bytes());
    buf.extend_from_slice(&pt.z.to_le_bytes());
}

/// Centers `mesh` at the origin and uniformly scales its vertices so the largest bounding-box axis
/// span equals `target_span_mm` millimeters (`5.0..=250.0 mm`).
pub fn scale_mesh_to_millimeters(mesh: &mut Mesh, target_span_mm: f32) -> Result<(), Error> {
    validate_mesh_geometry(mesh, true)?;
    if !target_span_mm.is_finite()
        || !(MIN_EXPORT_SPAN_MM..=MAX_EXPORT_SPAN_MM).contains(&target_span_mm)
    {
        return Err(invalid_mesh_error(
            "Target millimeter span must be finite within [5.0, 250.0] mm",
        ));
    }
    let mut min_pt = Point3D::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max_pt = Point3D::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in &mesh.vertices {
        min_pt.x = min_pt.x.min(v.x);
        min_pt.y = min_pt.y.min(v.y);
        min_pt.z = min_pt.z.min(v.z);
        max_pt.x = max_pt.x.max(v.x);
        max_pt.y = max_pt.y.max(v.y);
        max_pt.z = max_pt.z.max(v.z);
    }
    let span_x = f64::from(max_pt.x) - f64::from(min_pt.x);
    let span_y = f64::from(max_pt.y) - f64::from(min_pt.y);
    let span_z = f64::from(max_pt.z) - f64::from(min_pt.z);
    let max_span = span_x.max(span_y).max(span_z);
    if !max_span.is_finite() || max_span <= f64::from(Point3D::EPSILON) {
        return Err(invalid_mesh_error(
            "Mesh bounding box has degenerate zero span",
        ));
    }
    let cx = 0.5 * (f64::from(min_pt.x) + f64::from(max_pt.x));
    let cy = 0.5 * (f64::from(min_pt.y) + f64::from(max_pt.y));
    let cz = 0.5 * (f64::from(min_pt.z) + f64::from(max_pt.z));
    let scale = f64::from(target_span_mm) / max_span;
    for v in &mut mesh.vertices {
        v.x = ((f64::from(v.x) - cx) * scale) as f32;
        v.y = ((f64::from(v.y) - cy) * scale) as f32;
        v.z = ((f64::from(v.z) - cz) * scale) as f32;
    }
    Ok(())
}

/// Scales a copy of `mesh` to `target_span_mm` millimeters (`5.0..=250.0 mm`) and serializes
/// binary little-endian `.stl` bytes (`84 + 50 * F`) with `UNITS=mm` in the 80-byte header.
pub fn export_mesh_to_stl_binary_scaled(
    mesh: &Mesh,
    target_span_mm: f32,
) -> Result<Vec<u8>, Error> {
    let calibrated = MillimeterMesh::from_scaled(mesh.clone(), target_span_mm)?;
    export_mesh_to_stl_binary(&calibrated)
}

/// Serializes an indexed triangle mesh into standard binary little-endian `.stl` bytes (`84 + 50 * F`).
///
/// Stamps `UNITS=mm` exclusively when given a calibrated [`MillimeterMesh`]; unscaled [`Mesh`]
/// inputs emit [`STL_UNITLESS_HEADER_TEXT`] without claiming millimeter coordinates.
pub fn export_mesh_to_stl_binary<S: StlBinarySource + ?Sized>(
    source: &S,
) -> Result<Vec<u8>, Error> {
    let (mesh, header_text) = source.stl_mesh_and_header(StlBinarySeal(()));
    validate_mesh_geometry(mesh, true)?;
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let face_count = u32::try_from(triangles.len())
        .map_err(|_| invalid_mesh_error("STL triangle count exceeds u32::MAX"))?;

    let mut out = Vec::with_capacity(84 + 50 * triangles.len());
    let mut header = [0_u8; 80];
    let copy_len = header_text.len().min(header.len());
    header[..copy_len].copy_from_slice(&header_text[..copy_len]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&face_count.to_le_bytes());

    for &[i0, i1, i2] in triangles {
        let normal = resolve_triangle_normal(mesh, i0, i1, i2);
        write_point_le(&mut out, normal);
        write_point_le(&mut out, mesh.vertices[i0]);
        write_point_le(&mut out, mesh.vertices[i1]);
        write_point_le(&mut out, mesh.vertices[i2]);
        out.extend_from_slice(&0_u16.to_le_bytes());
    }

    Ok(out)
}

/// Serializes an indexed triangle [`Mesh`] into human-readable ASCII `.stl` text.
pub fn export_mesh_to_stl_ascii(mesh: &Mesh) -> Result<String, Error> {
    validate_mesh_geometry(mesh, true)?;
    let (triangles, _) = mesh.indices.as_chunks::<3>();
    let mut out = String::with_capacity(64 + triangles.len() * 180);
    out.push_str("solid oxidize_mesh\n");

    for &[i0, i1, i2] in triangles {
        let n = resolve_triangle_normal(mesh, i0, i1, i2);
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];
        let _ = writeln!(out, "  facet normal {:.6} {:.6} {:.6}", n.x, n.y, n.z);
        out.push_str("    outer loop\n");
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v0.x, v0.y, v0.z);
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v1.x, v1.y, v1.z);
        let _ = writeln!(out, "      vertex {:.6} {:.6} {:.6}", v2.x, v2.y, v2.z);
        out.push_str("    endloop\n  endfacet\n");
    }

    out.push_str("endsolid oxidize_mesh\n");
    Ok(out)
}

fn parse_triplet(tokens: &[&str], label: &str) -> Result<Point3D, Error> {
    if tokens.len() != 3 {
        return Err(invalid_mesh_error(&format!(
            "Expected 3 coordinates for {label}"
        )));
    }
    let x: f32 = tokens[0]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float X"))?;
    let y: f32 = tokens[1]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float Y"))?;
    let z: f32 = tokens[2]
        .parse()
        .map_err(|_| invalid_mesh_error("Invalid STL float Z"))?;
    let pt = Point3D::new(x, y, z);
    if !pt.is_finite() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Non-finite coordinate in ASCII STL",
        ));
    }
    Ok(pt)
}

/// Parses an ASCII `.stl` document back into a triangle soup [`Mesh`].
pub fn parse_stl_ascii(stl: &str) -> Result<Mesh, Error> {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut current_normal = Point3D::new(0.0, 0.0, 1.0);
    let mut facet_verts: Vec<Point3D> = Vec::with_capacity(3);
    let mut saw_solid = false;

    for raw_line in stl.lines() {
        let parts: Vec<&str> = raw_line.split_whitespace().collect();
        match parts.as_slice() {
            ["solid", ..] => saw_solid = true,
            ["facet", "normal", rest @ ..] => {
                current_normal = parse_triplet(rest, "facet normal")?;
                facet_verts.clear();
            }
            ["vertex", rest @ ..] => {
                facet_verts.push(parse_triplet(rest, "vertex")?);
            }
            ["endfacet"] => {
                if facet_verts.len() != 3 {
                    return Err(invalid_mesh_error("ASCII STL facet must have 3 vertices"));
                }
                let base = vertices.len();
                vertices.extend_from_slice(&facet_verts);
                normals.extend_from_slice(&[current_normal; 3]);
                indices.extend_from_slice(&[base, base + 1, base + 2]);
                facet_verts.clear();
            }
            _ => {}
        }
    }

    if !saw_solid || indices.is_empty() {
        return Err(invalid_mesh_error("ASCII STL contained no valid facets"));
    }
    Ok(Mesh {
        vertices,
        normals,
        indices,
    })
}
