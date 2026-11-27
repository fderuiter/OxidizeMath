//! Triangulated three-dimensional surface mesh primitive.

/// Triangulated three-dimensional cortical or geometric surface mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Vertex coordinates `[x, y, z]` in millimeters.
    pub vertices: Vec<[f64; 3]>,
    /// Outward-oriented triangle face vertex indices `[i0, i1, i2]`.
    pub faces: Vec<[usize; 3]>,
}

impl Surface {
    /// Constructs a triangulated [`Surface`] from vertex coordinates and triangle indices.
    #[must_use]
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    /// Constructs a vertex-only [`Surface`] with an empty triangle index list.
    #[must_use]
    pub fn from_vertices(vertices: Vec<[f64; 3]>) -> Self {
        Self {
            vertices,
            faces: Vec::new(),
        }
    }
}
