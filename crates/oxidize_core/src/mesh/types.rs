use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3D point or vector with single-precision floating point coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Default)]
#[allow(missing_docs)]
pub struct Mesh {
    #[allow(missing_docs)]
    pub vertices: Vec<Point3D>,
    #[allow(missing_docs)]
    pub normals: Vec<Point3D>,
    #[allow(missing_docs)]
    pub indices: Vec<usize>,
}

impl Mesh {
    /// Creates an empty [`Mesh`] without heap allocation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocates vertex/normal and index buffers for zero-allocation reuse.
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
