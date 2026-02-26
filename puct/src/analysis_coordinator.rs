use std::collections::HashSet;

use common::TranspositionHash;
use crossbeam::channel;
use model::GameAnalyzer;

type HashId = u64;
type SimId = usize;

type AnalyzeMsg<S> = (SimId, S);
type AnalyzeTx<S> = channel::Sender<AnalyzeMsg<S>>;
type AnalyzeRx<S> = channel::Receiver<AnalyzeMsg<S>>;

type CompleteTx = channel::Sender<HashId>;
type CompleteRx = channel::Receiver<HashId>;

type AnalyzerState<M> = <M as GameAnalyzer>::State;

pub(crate) trait InFlightExpansions: Send + Sync {
    type State;

    fn analyze(&self, sim_id: SimId, game_state: Self::State);

    fn complete(&self, hash: HashId);
}

pub(crate) struct InFlightExpansionsHandle<S> {
    analyze_tx: AnalyzeTx<S>,
    complete_tx: CompleteTx,
}

impl<S> Clone for InFlightExpansionsHandle<S> {
    fn clone(&self) -> Self {
        Self {
            analyze_tx: self.analyze_tx.clone(),
            complete_tx: self.complete_tx.clone(),
        }
    }
}

impl<S> InFlightExpansions for InFlightExpansionsHandle<S>
where
    S: Send,
{
    type State = S;

    fn analyze(&self, sim_id: SimId, game_state: S) {
        let _ = self.analyze_tx.send((sim_id, game_state));
    }

    fn complete(&self, hash: HashId) {
        let _ = self.complete_tx.send(hash);
    }
}

type CoordinatorPair<'a, M> = (
    InFlightExpansionsHandle<AnalyzerState<M>>,
    AnalysisCoordinator<'a, M>,
);

pub(crate) struct AnalysisCoordinator<'a, M>
where
    M: GameAnalyzer<RequestId = usize>,
{
    analyzer: &'a M,
    analyze_rx: AnalyzeRx<AnalyzerState<M>>,
    complete_rx: CompleteRx,
    in_flight: HashSet<HashId>,
}

impl<'a, M> AnalysisCoordinator<'a, M>
where
    M: GameAnalyzer<RequestId = usize>,
    AnalyzerState<M>: TranspositionHash,
{
    pub(crate) fn new(analyzer: &'a M, capacity: usize) -> CoordinatorPair<'a, M> {
        let (analyze_tx, analyze_rx) = channel::bounded::<AnalyzeMsg<AnalyzerState<M>>>(capacity);
        let (complete_tx, complete_rx) = channel::bounded::<HashId>(capacity);

        (
            InFlightExpansionsHandle {
                analyze_tx,
                complete_tx,
            },
            Self {
                analyzer,
                analyze_rx,
                complete_rx,
                in_flight: HashSet::new(),
            },
        )
    }

    pub(crate) fn run(mut self) {
        loop {
            while let Ok((sim_id, game_state)) = self.analyze_rx.try_recv() {
                self.maybe_analyze(sim_id, game_state);
            }

            if let Ok(hash) = self.complete_rx.try_recv() {
                self.in_flight.remove(&hash);
                continue;
            }

            match self.analyze_rx.recv() {
                Ok((sim_id, game_state)) => self.maybe_analyze(sim_id, game_state),
                Err(_) => break,
            }
        }
    }

    fn maybe_analyze(&mut self, sim_id: SimId, game_state: AnalyzerState<M>) {
        let hash = game_state.transposition_hash();
        if self.in_flight.insert(hash) {
            self.analyzer.analyze(sim_id, &game_state);
        }
    }
}
