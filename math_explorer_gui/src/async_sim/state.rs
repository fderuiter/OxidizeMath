use crate::framework::BrushStroke;
use std::sync::Arc;

/// Persisted spatial sculpt payload synchronized between `UnifiedSimTool` and worker models.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SculptStatePayload {
    /// RLE `[value, run_len]` pairs for 2D raster obstacle/cell masks.
    pub rle_mask: Vec<[u32; 2]>,
    /// Sparse `(index, value)` deltas for 1D potentials.
    pub sparse_potential: Vec<(u16, f64)>,
}

#[allow(missing_docs)]
pub enum SimCommand {
    #[allow(missing_docs)]
    Start,
    #[allow(missing_docs)]
    Pause,
    #[allow(missing_docs)]
    Step(usize),
    /// Rewinds the simulation by the requested step count via bounded checkpoint replay.
    StepBack(usize),
    /// Seeks deterministically to an exact step index within the retained checkpoint window.
    SeekStep(usize),
    #[allow(missing_docs)]
    Reset,
    #[allow(missing_docs)]
    SetSpeed(usize),
    #[allow(missing_docs)]
    TrainEpisodes(usize),
    #[allow(missing_docs)]
    ApplyBrush {
        #[allow(missing_docs)]
        cx: i32,
        #[allow(missing_docs)]
        cy: i32,
        #[allow(missing_docs)]
        r: i32,
        #[allow(missing_docs)]
        is_obstacle: bool,
    },
    /// Applies an interpolated spatial brush stroke to the simulation field.
    ApplyBrushStroke(BrushStroke),
    /// Synchronizes persisted spatial sculpt state with the worker model.
    SyncSculptState(SculptStatePayload),
    #[allow(missing_docs)]
    ClearObstacles,
    #[allow(missing_docs)]
    Custom(String),
}

impl SimCommand {
    #[allow(missing_docs)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Start => "Start",
            Self::Pause => "Pause",
            Self::Step(_) => "Step",
            Self::StepBack(_) => "StepBack",
            Self::SeekStep(_) => "SeekStep",
            Self::Reset => "Reset",
            Self::SetSpeed(_) => "SetSpeed",
            Self::TrainEpisodes(_) => "TrainEpisodes",
            Self::ApplyBrush { .. } => "ApplyBrush",
            Self::ApplyBrushStroke(_) => "ApplyBrushStroke",
            Self::SyncSculptState(_) => "SyncSculptState",
            Self::ClearObstacles => "ClearObstacles",
            Self::Custom(_) => "Custom",
        }
    }
}

#[allow(missing_docs)]
pub struct StateSnapshot {
    #[allow(missing_docs)]
    pub width: usize,
    #[allow(missing_docs)]
    pub height: usize,
    #[allow(missing_docs)]
    pub pixels: Arc<std::sync::RwLock<Vec<eframe::egui::Color32>>>,
    #[allow(missing_docs)]
    pub custom_data: Vec<f64>,
    #[allow(missing_docs)]
    pub structured_data: Option<Box<dyn std::any::Any + Send>>,
    #[allow(missing_docs)]
    pub sonification: crate::audio::SonificationFrame,
}

#[allow(missing_docs)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum SimWorkerState {
    #[default]
    Uninitialized,
    Ready,
    Running,
    Stepping,
    Faulted,
    Terminated,
}

impl SimWorkerState {
    #[allow(missing_docs)]
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    #[allow(missing_docs)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    #[allow(missing_docs)]
    pub fn is_stepping(&self) -> bool {
        matches!(self, Self::Stepping)
    }

    #[allow(missing_docs)]
    pub fn is_faulted(&self) -> bool {
        matches!(self, Self::Faulted)
    }

    #[allow(missing_docs)]
    pub fn is_terminated(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    #[allow(missing_docs)]
    pub fn is_command_valid(&self, cmd: &SimCommand) -> bool {
        match self {
            Self::Ready => !matches!(cmd, SimCommand::Pause),
            Self::Running => !matches!(
                cmd,
                SimCommand::Start
                    | SimCommand::Step(_)
                    | SimCommand::StepBack(_)
                    | SimCommand::TrainEpisodes(_)
            ),
            Self::Stepping => !matches!(
                cmd,
                SimCommand::Start
                    | SimCommand::Pause
                    | SimCommand::Step(_)
                    | SimCommand::StepBack(_)
                    | SimCommand::TrainEpisodes(_)
            ),
            Self::Faulted => matches!(cmd, SimCommand::Reset),
            Self::Uninitialized | Self::Terminated => false,
        }
    }
}

impl std::fmt::Display for SimWorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uninitialized => write!(f, "Uninitialized"),
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Stepping => write!(f, "Stepping"),
            Self::Faulted => write!(f, "Faulted"),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatusEvent {
    Ready,
    Running,
    Stepping,
    Paused,
    Faulted(String),
    Terminated,
}

impl WorkerStatusEvent {
    #[allow(missing_docs)]
    pub fn to_state(&self) -> SimWorkerState {
        match self {
            Self::Ready | Self::Paused => SimWorkerState::Ready,
            Self::Running => SimWorkerState::Running,
            Self::Stepping => SimWorkerState::Stepping,
            Self::Faulted(_) => SimWorkerState::Faulted,
            Self::Terminated => SimWorkerState::Terminated,
        }
    }
}

impl From<&WorkerStatusEvent> for SimWorkerState {
    fn from(event: &WorkerStatusEvent) -> Self {
        event.to_state()
    }
}

impl From<WorkerStatusEvent> for SimWorkerState {
    fn from(event: WorkerStatusEvent) -> Self {
        event.to_state()
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimCommandError {
    InvalidTransition {
        current_state: SimWorkerState,
        command: &'static str,
    },
    ChannelDisconnected,
}

impl std::fmt::Display for SimCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition {
                current_state,
                command,
            } => {
                write!(
                    f,
                    "Invalid simulation command '{command}' in state {current_state}"
                )
            }
            Self::ChannelDisconnected => {
                write!(f, "Simulation worker channel disconnected")
            }
        }
    }
}

impl std::error::Error for SimCommandError {}
