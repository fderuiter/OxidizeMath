//! Triangulated three-dimensional surface mesh primitive.

/// Triangulated three-dimensional surface mesh $\mathcal{M} = (\mathcal{V}, \mathcal{F})$ in $\mathbb{R}^3$.
///
/// Represents piecewise-linear 2-manifolds with vertex coordinates $\mathbf{v}_i \in \mathbb{R}^3$
/// and oriented triangle faces $(i_0, i_1, i_2) \in \mathcal{F}$ for symmetric Hausdorff/cortical
/// distance queries and discrete differential-geometry operators:
///
/// $$T(v_w, v_p) = \frac{1}{2}\left(\min_{\mathbf{u} \in \mathcal{V}_p} \|\mathbf{v}_w - \mathbf{u}\|_2 + \min_{\mathbf{w} \in \mathcal{V}_w} \|\mathbf{v}_p - \mathbf{w}\|_2\right)$$
///
/// Reference: Fischl & Dale (2000), *PNAS* 97(20), 11050–11055 (`freesurfer.tex`); see
/// [`docs/math_specs/pure_math.md`](../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Stores contiguous vertex and triangle index arrays suitable for zero-allocation CSR spatial
/// indexing via [`super::FlatSpatialGrid`] and [`super::ThicknessWorkspace`].
///
/// # Domain of Validity
/// Finite 3D vertex coordinates `[x, y, z]` and valid zero-based vertex indices `faces[f][k] < vertices.len()`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Surface {
    /// Vertex coordinates `[x, y, z]` in millimeters (or world units).
    pub vertices: Vec<[f64; 3]>,
    /// Outward-oriented triangle face vertex indices `[i0, i1, i2]`.
    pub faces: Vec<[usize; 3]>,
}

impl Surface {
    /// Constructs a triangulated [`Surface`] from vertex coordinates and triangle indices.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    /// Constructs a vertex-only [`Surface`] with an empty triangle index list.
    #[must_use]
    #[verified_engine::verified]
    pub fn from_vertices(vertices: Vec<[f64; 3]>) -> Self {
        Self {
            vertices,
            faces: Vec::new(),
        }
    }
}
