//! Error types for `pure_math::graph_theory` solvers and topology interchange codecs.

use thiserror::Error;

/// Errors returned by network flow solvers, spanning-tree routines, and topology interchange codecs.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GraphTheoryError {
    /// Requested source or sink terminal index is outside `0..vertex_count`.
    #[error("terminal vertex {node} is out of bounds for graph with {vertex_count} vertices")]
    InvalidTerminal {
        /// Out-of-bounds terminal vertex index.
        node: usize,
        /// Total vertex count in the graph.
        vertex_count: usize,
    },
    /// Source and sink terminals must be distinct vertices in an `s-t` flow problem.
    #[error("source and sink terminals must be distinct (got vertex {node})")]
    SourceEqualsSink {
        /// Duplicated terminal vertex index.
        node: usize,
    },
    /// Edge capacity must be finite and non-negative.
    #[error("edge {edge_index} has invalid capacity {value} (must be finite and >= 0)")]
    InvalidCapacity {
        /// Zero-based index of the invalid edge.
        edge_index: usize,
        /// Invalid capacity value.
        value: f64,
    },
    /// Edge weight must be finite (`!is_nan() && !is_infinite()`).
    #[error("edge weight must be finite (got {value})")]
    InvalidWeight {
        /// Non-finite weight value.
        value: f64,
    },
    /// Vertex 2D layout coordinate must be finite.
    #[error("vertex {node} has non-finite coordinate ({x}, {y})")]
    InvalidCoordinate {
        /// Vertex identifier with invalid coordinates.
        node: usize,
        /// X coordinate.
        x: f64,
        /// Y coordinate.
        y: f64,
    },
    /// Preallocated output slice passed to an in-place solver has the wrong length.
    #[error("output slice length mismatch (expected {expected}, got {actual})")]
    ScratchBufferMismatch {
        /// Expected slice length.
        expected: usize,
        /// Actual slice length provided by the caller.
        actual: usize,
    },
    /// Imported or serialized topology exceeds [`super::interchange::MAX_INTERCHANGE_VERTICES`].
    #[error("vertex count {count} exceeds maximum limit of {max}")]
    VertexLimitExceeded {
        /// Encountered vertex count.
        count: usize,
        /// Maximum permitted vertex count.
        max: usize,
    },
    /// Imported or serialized topology exceeds [`super::interchange::MAX_INTERCHANGE_EDGES`].
    #[error("edge count {count} exceeds maximum limit of {max}")]
    EdgeLimitExceeded {
        /// Encountered edge count.
        count: usize,
        /// Maximum permitted edge count.
        max: usize,
    },
    /// Edge endpoint references an undeclared vertex identifier.
    #[error("edge references undeclared endpoint vertex {node}")]
    UnknownEndpoint {
        /// Missing vertex identifier.
        node: usize,
    },
    /// Syntax or structural parse failure in `.dot`, `.graphml`, or `.csv` interchange text.
    #[error("failed to parse {format} topology: {message}")]
    ParseError {
        /// Format identifier (`"DOT"`, `"GraphML"`, or `"CSV"`).
        format: &'static str,
        /// Human-readable error message.
        message: String,
    },
    /// Imported topology byte length exceeds [`super::interchange::MAX_TOPOLOGY_IMPORT_BYTES`].
    #[error("topology payload ({actual} bytes) exceeds maximum limit of {max} bytes")]
    PayloadTooLarge {
        /// Observed UTF-8 byte length.
        actual: usize,
        /// Maximum permitted UTF-8 byte length.
        max: usize,
    },
    /// Untrusted GraphML XML contained a forbidden `<!DOCTYPE` or `<!ENTITY` declaration.
    #[error("XML DOCTYPE and ENTITY declarations are forbidden in untrusted GraphML inputs")]
    ForbiddenXmlDeclaration,
}

// [cite:graph_parameters_rust]
