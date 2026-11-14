use super::SimCommand;

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
            Self::Running => !matches!(cmd, SimCommand::Start | SimCommand::TrainEpisodes(_)),
            Self::Stepping => !matches!(
                cmd,
                SimCommand::Start | SimCommand::Pause | SimCommand::TrainEpisodes(_)
            ),
            Self::Uninitialized | Self::Faulted | Self::Terminated => false,
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
