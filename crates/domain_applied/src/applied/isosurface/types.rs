pub use oxidize_core::mesh::{Mesh, Point3D};

/// A 3D grid of scalar values, used as input for isosurface extraction.
///
/// The grid data is stored in a flat vector, indexed by `z * width * height + y * width + x`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for VoxelGrid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct VoxelGridRaw {
            width: usize,
            height: usize,
            depth: usize,
            data: Vec<f32>,
            voxel_size: Point3D,
            origin: Point3D,
        }
        let raw = VoxelGridRaw::deserialize(deserializer)?;
        if ![
            raw.voxel_size.x,
            raw.voxel_size.y,
            raw.voxel_size.z,
            raw.origin.x,
            raw.origin.y,
            raw.origin.z,
        ]
        .into_iter()
        .all(f32::is_finite)
        {
            return Err(serde::de::Error::custom(
                "VoxelGrid voxel_size and origin must be finite",
            ));
        }
        VoxelGrid::builder()
            .dimensions(raw.width, raw.height, raw.depth)
            .data(raw.data)
            .voxel_size(raw.voxel_size)
            .origin(raw.origin)
            .build()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
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

    /// Mutable access to the underlying flat scalar buffer (`z -> y -> x` order) for zero-allocation updates.
    #[verified_engine::verified]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Writes `value` at `(x, y, z)` if within grid bounds.
    #[inline]
    #[verified_engine::verified]
    pub fn set(&mut self, x: usize, y: usize, z: usize, value: f32) {
        if x < self.width && y < self.height && z < self.depth {
            let idx = (z * self.height + y) * self.width + x;
            if idx < self.data.len() {
                self.data[idx] = value;
            }
        }
    }

    /// Reconfigures dimensions, spacing, and origin in-place, reusing existing capacity when sufficient.
    #[verified_engine::verified]
    pub fn reconfigure_in_place(
        &mut self,
        dims: (usize, usize, usize),
        voxel_size: Point3D,
        origin: Point3D,
    ) -> Result<(), VoxelGridError> {
        let (width, height, depth) = dims;
        if width == 0 || height == 0 || depth == 0 {
            return Err(VoxelGridError::InvalidDimensions);
        }
        if voxel_size.x <= 0.0 || voxel_size.y <= 0.0 || voxel_size.z <= 0.0 {
            return Err(VoxelGridError::InvalidVoxelSize);
        }
        let total = width
            .checked_mul(height)
            .and_then(|wh| wh.checked_mul(depth))
            .ok_or(VoxelGridError::InvalidDimensions)?;
        self.width = width;
        self.height = height;
        self.depth = depth;
        self.voxel_size = voxel_size;
        self.origin = origin;
        self.data.resize(total, 0.0);
        Ok(())
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
