use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3D point or vector with single-precision floating point coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point3D {
    #[allow(missing_docs)]
    pub x: f32,
    #[allow(missing_docs)]
    pub y: f32,
    #[allow(missing_docs)]
    pub z: f32,
}

impl Point3D {
    #[allow(missing_docs)]
    pub const EPSILON: f32 = 1.0e-6;
    #[inline]
    #[allow(missing_docs)]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    #[inline]
    #[allow(missing_docs)]
    pub fn dot(&self, other: Point3D) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    #[inline]
    #[allow(missing_docs)]
    pub fn cross(&self, other: Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    #[inline]
    #[allow(missing_docs)]
    pub fn magnitude(&self) -> f32 {
        self.dot(*self).sqrt()
    }
    #[inline]
    #[allow(missing_docs)]
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

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct Mesh {
    #[allow(missing_docs)]
    pub vertices: Vec<Point3D>,
    #[allow(missing_docs)]
    pub normals: Vec<Point3D>,
    #[allow(missing_docs)]
    pub indices: Vec<usize>,
}

/// An 8-bit per-channel RGB color attribute for a 3D mesh vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
