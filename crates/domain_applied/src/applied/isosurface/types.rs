pub use oxidize_core::mesh::{Mesh, Point3D};

/// A 3D grid of scalar values, used as input for isosurface extraction.
///
/// The grid data is stored in a flat vector, indexed by `z * width * height + y * width + x`.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelGrid {
    /// Number of voxels along the X axis.
    width: usize,
    /// Number of voxels along the Y axis.
    height: usize,
    /// Number of voxels along the Z axis.
    depth: usize,
    /// Flat vector of scalar values (e.g., signed distance or density).
    /// Indexing is `z` (slowest) -> `y` -> `x` (fastest).
    data: Vec<f32>,
    /// The physical size of each voxel (dx, dy, dz).
    voxel_size: Point3D,
    /// The physical coordinate of the grid's origin (0,0,0 index).
    origin: Point3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum VoxelGridError {
    #[allow(missing_docs)]
    InvalidDimensions,
    #[allow(missing_docs)]
    DataSizeMismatch,
    #[allow(missing_docs)]
    InvalidVoxelSize,
}

impl std::fmt::Display for VoxelGridError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "Grid dimensions must be greater than zero"),
            Self::DataSizeMismatch => {
                write!(f, "Data vector size does not match width * height * depth")
            }
            Self::InvalidVoxelSize => write!(f, "Voxel size components must be strictly positive"),
        }
    }
}

impl std::error::Error for VoxelGridError {}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelGridBuilder {
    width: Option<usize>,
    height: Option<usize>,
    depth: Option<usize>,
    data: Option<Vec<f32>>,
    voxel_size: Point3D,
    origin: Point3D,
}

impl Default for VoxelGridBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelGridBuilder {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            width: None,
            height: None,
            depth: None,
            data: None,
            voxel_size: Point3D::new(1.0, 1.0, 1.0),
            origin: Point3D::new(0.0, 0.0, 0.0),
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn dimensions(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self.depth = Some(depth);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn data(mut self, data: Vec<f32>) -> Self {
        self.data = Some(data);
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn voxel_size(mut self, size: Point3D) -> Self {
        self.voxel_size = size;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn origin(mut self, origin: Point3D) -> Self {
        self.origin = origin;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn build(self) -> Result<VoxelGrid, VoxelGridError> {
        let width = self.width.ok_or(VoxelGridError::InvalidDimensions)?;
        let height = self.height.ok_or(VoxelGridError::InvalidDimensions)?;
        let depth = self.depth.ok_or(VoxelGridError::InvalidDimensions)?;
        let data = self.data.ok_or(VoxelGridError::DataSizeMismatch)?;

        if width == 0 || height == 0 || depth == 0 {
            return Err(VoxelGridError::InvalidDimensions);
        }

        if data.len() != width * height * depth {
            return Err(VoxelGridError::DataSizeMismatch);
        }

        if self.voxel_size.x <= 0.0 || self.voxel_size.y <= 0.0 || self.voxel_size.z <= 0.0 {
            return Err(VoxelGridError::InvalidVoxelSize);
        }

        Ok(VoxelGrid {
            width,
            height,
            depth,
            data,
            voxel_size: self.voxel_size,
            origin: self.origin,
        })
    }
}

impl VoxelGrid {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn builder() -> VoxelGridBuilder {
        VoxelGridBuilder::new()
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn width(&self) -> usize {
        self.width
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn height(&self) -> usize {
        self.height
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn depth(&self) -> usize {
        self.depth
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn voxel_size(&self) -> &Point3D {
        &self.voxel_size
    }
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn origin(&self) -> &Point3D {
        &self.origin
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_voxel_size(&mut self, size: Point3D) -> Result<(), VoxelGridError> {
        if size.x <= 0.0 || size.y <= 0.0 || size.z <= 0.0 {
            return Err(VoxelGridError::InvalidVoxelSize);
        }
        self.voxel_size = size;
        Ok(())
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn set_origin(&mut self, origin: Point3D) {
        self.origin = origin;
    }
    /// Retrieves the value at the given grid index (x, y, z).
    ///
    /// Returns `0.0` if the coordinates are out of bounds.
    #[inline]
    #[verified_engine::verified]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        if x >= self.width || y >= self.height || z >= self.depth {
            return 0.0; // Boundary condition
        }
        // Sentinel: Prevent panic if data buffer is malformed (smaller than dimensions imply)
        let idx = z
            .checked_mul(self.width)
            .and_then(|val| val.checked_mul(self.height))
            .and_then(|val| val.checked_add(y.checked_mul(self.width)?))
            .and_then(|val| val.checked_add(x));

        match idx {
            Some(i) => self.data.get(i).copied().unwrap_or(0.0),
            None => 0.0,
        }
    }
}
