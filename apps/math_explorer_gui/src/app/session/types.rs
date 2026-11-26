use crate::app::AppState;
use serde::{Deserialize, Serialize};

/// Explicit lifecycle states governing session persistence and history operations.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SessionLifecycleState {
    /// Resting state when no drag, restore, dialog, save, or load operation is active.
    #[default]
    Idle,
    /// Continuous slider interaction actively coalescing parameter edits.
    DragEditing,
    /// Settling multi-frame state updates during undo/redo or file restoration.
    RestoringHistory,
    /// Awaiting file picker selection from background dialog thread.
    AwaitingFileDialog,
    /// Asynchronously persisting workspace session payload to disk.
    Saving,
    /// Asynchronously loading and parsing workspace session payload.
    Loading,
    /// Encountered an invalid transition or I/O failure.
    Error,
}

impl SessionLifecycleState {
    /// Returns a plain-language description of the current lifecycle state for user-facing messages.
    pub fn user_label(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::DragEditing => "editing parameters",
            Self::RestoringHistory => "restoring history",
            Self::AwaitingFileDialog => "awaiting file selection",
            Self::Saving => "saving workspace",
            Self::Loading => "loading workspace",
            Self::Error => "in an error state",
        }
    }
}

impl std::fmt::Display for SessionLifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::DragEditing => write!(f, "DragEditing"),
            Self::RestoringHistory => write!(f, "RestoringHistory"),
            Self::AwaitingFileDialog => write!(f, "AwaitingFileDialog"),
            Self::Saving => write!(f, "Saving"),
            Self::Loading => write!(f, "Loading"),
            Self::Error => write!(f, "Error"),
        }
    }
}

/// Metadata envelope wrapping application state for versioned workspace session persistence.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkspaceEnvelope {
    /// Schema version for envelope serialization format (e.g. "1.0").
    pub schema_version: String,
    /// Application version emitting the envelope.
    pub app_version: String,
    /// ISO 8601 creation or export timestamp string.
    pub timestamp: String,
    /// Wrapped application state payload.
    pub app_state: AppState,
    /// Recent undo history snapshots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history_undo_stack: Vec<AppState>,
    /// Recent redo history snapshots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history_redo_stack: Vec<AppState>,
    /// Persisted plot callout annotations.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "callouts")]
    pub callout_manager: Option<crate::framework::CalloutManager>,
}
