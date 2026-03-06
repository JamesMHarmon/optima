use crate::search_context::{PathStep, SearchContextGuard};

pub(super) enum SelectionResult<S, T> {
    Terminal(TerminalSelectionResult<T>),
    Leaf(LeafSelectionResult<S>),
    Preempted(PreemptedSelectionResult),
}

impl<S, T> SelectionResult<S, T> {
    pub(super) fn new_terminal(context: SearchContextGuard, terminal: T, depth: usize) -> Self {
        Self::Terminal(TerminalSelectionResult::new(context, terminal, depth))
    }

    pub(super) fn new_leaf(context: SearchContextGuard, game_state: S, depth: usize) -> Self {
        Self::Leaf(LeafSelectionResult::new(context, game_state, depth))
    }

    pub(super) fn new_preempted(context: SearchContextGuard, depth: usize) -> Self {
        Self::Preempted(PreemptedSelectionResult::new(context, depth))
    }
}

pub(super) struct TerminalSelectionResult<T> {
    context: SearchContextGuard,
    pub(super) terminal: T,
    pub(super) depth: usize,
}

impl<T> TerminalSelectionResult<T> {
    fn new(context: SearchContextGuard, terminal: T, depth: usize) -> Self {
        Self {
            context,
            terminal,
            depth,
        }
    }

    pub(super) fn path(&self) -> &[PathStep] {
        &self.context.get_ref().path
    }
}

pub(super) struct LeafSelectionResult<S> {
    context: SearchContextGuard,
    pub(super) game_state: S,
    pub(super) depth: usize,
}

impl<S> LeafSelectionResult<S> {
    fn new(context: SearchContextGuard, game_state: S, depth: usize) -> Self {
        Self {
            context,
            game_state,
            depth,
        }
    }

    pub(super) fn path(&self) -> &[PathStep] {
        &self.context.get_ref().path
    }
}

pub(super) struct PreemptedSelectionResult {
    context: SearchContextGuard,
    pub(super) depth: usize,
}

impl PreemptedSelectionResult {
    fn new(context: SearchContextGuard, depth: usize) -> Self {
        Self { context, depth }
    }

    pub(super) fn path(&self) -> &[PathStep] {
        &self.context.get_ref().path
    }
}

/// Describes the outcome of one simulation step.
pub(super) enum SimulationStep<S, T> {
    /// The leaf was a terminal state
    Terminal(TerminalStep<T>),
    /// A previously-unseen position was reached and claimed by this sim.
    NewLeaf(NewLeafStep<S>),
    /// This sim was preempted — another sim already claimed this leaf position.
    Preempted(PreemptedStep),
}

impl<S, T> SimulationStep<S, T> {
    pub(super) fn depth(&self) -> usize {
        match self {
            Self::Terminal(s) => s.depth,
            Self::NewLeaf(s) => s.depth,
            Self::Preempted(s) => s.depth,
        }
    }

    pub(super) fn new_terminal(
        sim_id: usize,
        path: Vec<PathStep>,
        terminal: T,
        depth: usize,
    ) -> Self {
        SimulationStep::Terminal(TerminalStep::new(sim_id, path, terminal, depth))
    }

    pub(super) fn new_leaf(
        sim_id: usize,
        path: Vec<PathStep>,
        game_state: S,
        depth: usize,
    ) -> Self {
        SimulationStep::NewLeaf(NewLeafStep::new(sim_id, path, game_state, depth))
    }

    pub(super) fn new_preempted(sim_id: usize, path: Vec<PathStep>, depth: usize) -> Self {
        SimulationStep::Preempted(PreemptedStep::new(sim_id, path, depth))
    }
}

pub(super) struct TerminalStep<T> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) terminal: T,
    pub(super) depth: usize,
}

impl<T> TerminalStep<T> {
    fn new(sim_id: usize, path: Vec<PathStep>, terminal: T, depth: usize) -> Self {
        Self {
            sim_id,
            path,
            terminal,
            depth,
        }
    }
}

pub(super) struct NewLeafStep<S> {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) game_state: S,
    pub(super) depth: usize,
}

impl<S> NewLeafStep<S> {
    fn new(sim_id: usize, path: Vec<PathStep>, game_state: S, depth: usize) -> Self {
        Self {
            sim_id,
            path,
            game_state,
            depth,
        }
    }
}

pub(super) struct PreemptedStep {
    pub(super) sim_id: usize,
    pub(super) path: Vec<PathStep>,
    pub(super) depth: usize,
}

impl PreemptedStep {
    fn new(sim_id: usize, path: Vec<PathStep>, depth: usize) -> Self {
        Self {
            sim_id,
            path,
            depth,
        }
    }
}
