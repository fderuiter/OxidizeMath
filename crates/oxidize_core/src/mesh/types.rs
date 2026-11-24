use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3D point or vector with single-precision floating point coordinates.
///
/// # Examples
/// ```
/// use oxidize_core::mesh::Point3D;
///
/// let a = Point3D::new(1.0, 0.0, 0.0);
/// let b = Point3D::new(0.0, 1.0, 0.0);
/// assert_eq!(a.cross(b), Point3D::new(0.0, 0.0, 1.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// Cartesian $x$ coordinate.
    pub x: f32,
    /// Cartesian $y$ coordinate.
    pub y: f32,
    /// Cartesian $z$ coordinate.
    pub z: f32,
}

impl Point3D {
    /// Minimum vector norm threshold below which [`Point3D::normalize`] returns the zero vector.
    pub const EPSILON: f32 = 1.0e-6;
    /// Constructs a 3D point `(x, y, z)`.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Computes the Euclidean dot product $\mathbf{u} \cdot \mathbf{v}$.
    #[inline]
    pub fn dot(&self, other: Point3D) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    /// Computes the right-handed 3D cross product $\mathbf{u} \times \mathbf{v}$.
    #[inline]
    pub fn cross(&self, other: Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    /// Computes the Euclidean norm $\|\mathbf{u}\|_2$.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.dot(*self).sqrt()
    }
    /// Returns the unit vector $\mathbf{u} / \|\mathbf{u}\|_2$, or `(0, 0, 0)` when $\|\mathbf{u}\|_2 \le \epsilon$.
    #[inline]
    pub fn normalize(&self) -> Point3D {
        let mag = self.magnitude();
        if mag > Self::EPSILON {
            *self / mag
        } else {
            Point3D::new(0.0, 0.0, 0.0)
        }
    }
    /// Returns `true` when all three coordinates are finite numbers.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Point3D {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl Sub for Point3D {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl Neg for Point3D {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f32> for Point3D {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}
impl Mul<Point3D> for f32 {
    type Output = Point3D;
    #[inline]
    fn mul(self, point: Point3D) -> Point3D {
        point * self
    }
}
impl Div<f32> for Point3D {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self::new(self.x / scalar, self.y / scalar, self.z / scalar)
    }
}

/// Indexed 3D triangle mesh storing vertex positions, optional vertex normals, and triangle index triplets.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    /// Vertex positions in 3D space.
    pub vertices: Vec<Point3D>,
    /// Per-vertex normal vectors (optional or aligned with `vertices`).
    pub normals: Vec<Point3D>,
    /// Flat triangle index buffer (`indices.len()` is a multiple of 3).
    pub indices: Vec<usize>,
}

impl Mesh {
    /// Creates an empty [`Mesh`] without heap allocation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocates vertex/normal and index buffers for zero-allocation reuse.
    #[must_use]
    pub fn with_capacity(vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_capacity),
            normals: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(index_capacity),
        }
    }

    /// Clears vertices, normals, and triangle indices while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.normals.clear();
        self.indices.clear();
    }
}

/// An 8-bit per-channel RGB color attribute for a 3D mesh vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VertexColor {
    /// Red channel (`0..=255`).
    pub r: u8,
    /// Green channel (`0..=255`).
    pub g: u8,
    /// Blue channel (`0..=255`).
    pub b: u8,
}

impl VertexColor {
    /// Constructs a new RGB vertex color.
    #[inline]
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// An indexed 3D triangle mesh paired with per-vertex RGB color attributes.
#[derive(Debug, Clone, Default)]
pub struct ColoredMesh {
    /// Indexed triangle geometry and normals.
    pub mesh: Mesh,
    /// Per-vertex RGB colors aligned with `mesh.vertices`.
    pub colors: Vec<VertexColor>,
}

impl ColoredMesh {
    /// Constructs a [`ColoredMesh`] from an indexed [`Mesh`] and per-vertex colors.
    #[must_use]
    pub fn new(mesh: Mesh, colors: Vec<VertexColor>) -> Self {
        Self { mesh, colors }
    }
}
