use common::{GameLength, PlayerValue};

use super::*;

type TestPolicy = MovesLeftSelectionPolicy<ConstantCpuct, TestState, NoTrajectoryTerminal<u8, ()>>;

type Scenario<'a> = (
    Vec<EdgeInfo<'a, u8, MovesLeftSnapshot>>,
    u32,
    TestState,
    u32,
);

#[derive(Clone, Copy, Default)]
struct TestState {
    ptm: usize,
}

impl PlayerToMove for TestState {
    fn player_to_move(&self) -> usize {
        self.ptm
    }
}

#[derive(Clone, Copy)]
struct ConstantCpuct(f32);

impl CPUCT for ConstantCpuct {
    type State = TestState;

    fn cpuct(&self, _state: &Self::State, _nsb: u32, _is_root: bool) -> f32 {
        self.0
    }
}

fn snap(p1: f32, p2: f32, game_length: f32, weight: u32) -> MovesLeftSnapshot {
    MovesLeftSnapshot {
        p1_sum: (p1 as f64) * (weight as f64),
        p2_sum: (p2 as f64) * (weight as f64),
        game_length_sum: (game_length as f64) * (weight as f64),
        total_weight: weight,
    }
}

fn edge<'a, A>(
    edge_index: usize,
    action: &'a A,
    policy_prior: f32,
    visits: u32,
    snapshot: Option<MovesLeftSnapshot>,
) -> EdgeInfo<'a, A, MovesLeftSnapshot> {
    EdgeInfo {
        edge_index,
        action,
        policy_prior,
        visits,
        virtual_visits: 0,
        snapshot,
    }
}

fn edge_with_virtual_visits<'a, A>(
    edge_index: usize,
    action: &'a A,
    policy_prior: f32,
    visits: u32,
    virtual_visits: u32,
    snapshot: Option<MovesLeftSnapshot>,
) -> EdgeInfo<'a, A, MovesLeftSnapshot> {
    EdgeInfo {
        edge_index,
        action,
        policy_prior,
        visits,
        virtual_visits,
        snapshot,
    }
}

fn run_policy<'a, A, T>(
    policy: &MovesLeftSelectionPolicy<ConstantCpuct, TestState, T>,
    edges: &'a [EdgeInfo<'a, A, MovesLeftSnapshot>],
    node_visits: u32,
    state: &TestState,
    depth: u32,
) -> usize
where
    A: 'a,
    T: TrajectoryTerminal<TestState, Action = A>,
{
    policy.select_edge(
        NodeInfo {
            visits: node_visits,
            virtual_visits: 0,
            depth,
        },
        edges.iter().map(|e| EdgeInfo {
            edge_index: e.edge_index,
            action: e.action,
            policy_prior: e.policy_prior,
            visits: e.visits,
            virtual_visits: e.virtual_visits,
            snapshot: e.snapshot,
        }),
        state,
    )
}

fn reference_two_pass_select<'a, A>(
    edges: &[EdgeInfo<'a, A, MovesLeftSnapshot>],
    node_visits: u32,
    state: &TestState,
    depth: u32,
    cpuct: f32,
    options: &MovesLeftStrategyOptions,
) -> usize
where
    A: 'a,
{
    let is_root = depth == 0;
    let fpu = if is_root {
        options.fpu_root
    } else {
        options.fpu
    };

    let root_nsb = (node_visits as f32).sqrt();
    let player_to_move = state.player_to_move();

    let baseline = MovesLeftSelectionPolicy::<ConstantCpuct, TestState>::game_length_baseline(
        edges,
        options.moves_left_threshold,
        player_to_move,
    );

    let mut best_index = 0usize;
    let mut best_score = f32::MIN;

    for e in edges {
        let completed_visits = e.visits;
        let virtual_visits = e.virtual_visits;
        let nsa = completed_visits + virtual_visits;
        let psa = e.policy_prior;
        let usa = cpuct * psa * root_nsb / (1.0 + nsa as f32);

        let qsa_raw = e
            .snapshot
            .map(|s| s.player_value(player_to_move))
            .unwrap_or(fpu);

        let qsa = if nsa == 0 {
            qsa_raw
        } else {
            qsa_raw * (completed_visits as f32) / (nsa as f32)
        };

        let msa = MovesLeftSelectionPolicy::<ConstantCpuct, TestState>::msa(
            e.snapshot, &baseline, options,
        );
        let score = qsa + usa + msa;

        if score > best_score {
            best_score = score;
            best_index = e.edge_index;
        }
    }

    best_index
}

fn default_options() -> MovesLeftStrategyOptions {
    MovesLeftStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 1.0,
    }
}

#[test]
fn snapshot_value_and_length_are_weighted_means() {
    let s = snap(0.25, 0.75, 10.0, 4);
    assert!((s.player_value(1) - 0.25).abs() < 1e-6);
    assert!((s.player_value(2) - 0.75).abs() < 1e-6);
    assert!((s.game_length() - 10.0).abs() < 1e-6);
}

#[test]
fn weighted_merge_accumulates_sums_and_weight() {
    let mut a = MovesLeftSnapshot::zero();
    let b = snap(0.4, 0.6, 12.0, 1);

    a.merge_weighted(&b, 2);

    assert_eq!(a.total_weight, 2);
    assert!((a.player_value(1) - 0.4).abs() < 1e-6);
    assert!((a.player_value(2) - 0.6).abs() < 1e-6);
    assert!((a.game_length() - 12.0).abs() < 1e-6);
}

#[test]
fn weighted_merge_ignores_zero_weight_or_zero_other_weight() {
    let mut a = snap(0.3, 0.7, 5.0, 1);
    let b = snap(0.9, 0.1, 50.0, 1);
    a.merge_weighted(&b, 0);
    assert_eq!(a, snap(0.3, 0.7, 5.0, 1));

    let mut c = snap(0.3, 0.7, 5.0, 1);
    let d = snap(0.9, 0.1, 50.0, 0);
    c.merge_weighted(&d, 1);
    assert_eq!(c, snap(0.3, 0.7, 5.0, 1));
}

#[test]
fn threshold_ge_one_disables_moves_left_bias() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 1.0,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;

    let edges = [
        edge(0, &a0, 0.0, 5, Some(snap(0.2, 0.8, 5.0, 1))),
        edge(1, &a1, 0.0, 5, Some(snap(0.21, 0.79, 100.0, 1))),
    ];

    // With threshold >= 1, baseline is None so MSA = 0; choose best Q.
    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 1);
}

#[test]
fn fpu_root_used_at_depth_zero_and_fpu_used_below_root() {
    let options = MovesLeftStrategyOptions {
        fpu_root: 0.9,
        fpu: 0.1,
        ..default_options()
    };

    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;

    let edges = [
        edge(0, &a0, 0.0, 0, Some(snap(0.2, 0.8, 10.0, 1))),
        edge(1, &a1, 0.0, 0, None),
    ];

    // Root: missing snapshot uses fpu_root = 0.9, should win.
    let root_idx = run_policy(&policy, &edges, 1, &TestState { ptm: 1 }, 0);
    assert_eq!(root_idx, 1);

    // Non-root: missing snapshot uses fpu = 0.1, should lose.
    let child_idx = run_policy(&policy, &edges, 1, &TestState { ptm: 1 }, 1);
    assert_eq!(child_idx, 0);
}

#[test]
fn virtual_visits_down_weight_q_value() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 1.0,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;

    let edges = [
        edge_with_virtual_visits(0, &a0, 0.0, 10, 0, Some(snap(0.8, 0.2, 10.0, 1))),
        edge_with_virtual_visits(1, &a1, 0.0, 10, 5, Some(snap(0.8, 0.2, 10.0, 1))),
    ];

    // With no U term (cpuct = 0) and no MSA (threshold >= 1), selection is purely Q.
    // Virtual visits are treated as a loss of 0, so the second edge's Q is down-weighted.
    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 0);
}

#[test]
fn virtual_visits_do_not_reduce_u_term() {
    let options = MovesLeftStrategyOptions {
        // Disable moves-left bias so selection is Q + U.
        moves_left_threshold: 1.0,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(1.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;

    // Force Q to be exactly 0 for both edges so the selection depends only on U.
    // Virtual visits should reduce U by increasing the effective visit count.
    let edges = [
        edge_with_virtual_visits(0, &a0, 0.90, 1, 0, Some(snap(0.0, 1.0, 10.0, 1))),
        edge_with_virtual_visits(1, &a1, 1.00, 1, 100, Some(snap(0.0, 1.0, 10.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 0);
}

#[test]
fn winning_baseline_prefers_shorter_game_length() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 1.0,
        ..default_options()
    };

    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;

    // Baseline is most-visited explored edge (edge 0): Q=0.9 (winning), len=20.
    // Candidates have equal Q, differ only in length.
    let edges = [
        edge(0, &a0, 0.0, 10, Some(snap(0.9, 0.1, 20.0, 1))),
        edge(1, &a1, 0.0, 1, Some(snap(0.5, 0.5, 10.0, 1))),
        edge(2, &a2, 0.0, 1, Some(snap(0.5, 0.5, 30.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 1);
}

#[test]
fn losing_baseline_prefers_longer_game_length() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 1.0,
        ..default_options()
    };

    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;

    // Baseline is most-visited explored edge (edge 0): Q=0.1 (losing), len=20.
    // Candidates have equal Q, differ only in length.
    let edges = [
        edge(0, &a0, 0.0, 10, Some(snap(0.1, 0.9, 20.0, 1))),
        edge(1, &a1, 0.0, 1, Some(snap(0.5, 0.5, 10.0, 1))),
        edge(2, &a2, 0.0, 1, Some(snap(0.5, 0.5, 30.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 2);
}

#[test]
fn uncertain_baseline_does_not_apply_moves_left_bias() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 10.0,
        ..default_options()
    };

    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;

    // Baseline Q=0.5 is between [0.3,0.7], so no directive.
    // Candidate 2 has higher base Q but worse length; should still win.
    let edges = [
        edge(0, &a0, 0.0, 10, Some(snap(0.50, 0.50, 20.0, 1))),
        edge(1, &a1, 0.0, 1, Some(snap(0.55, 0.45, 10.0, 1))),
        edge(2, &a2, 0.0, 1, Some(snap(0.56, 0.44, 30.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx, 2);
}

#[test]
fn baseline_is_most_visited_explored_edge() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 2.0,
        ..default_options()
    };

    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let b0 = 0u8;
    let b1 = 1u8;
    let c0 = 2u8;
    let c1 = 3u8;

    // Case 1: most-visited baseline is uncertain => no directive => long wins by Q.
    let edges_uncertain = [
        edge(0, &b0, 0.0, 10, Some(snap(0.39, 0.61, 20.0, 1))),
        edge(1, &b1, 0.0, 9, Some(snap(0.20, 0.80, 20.0, 1))),
        edge(2, &c0, 0.0, 1, Some(snap(0.93, 0.07, 5.0, 1))),
        edge(3, &c1, 0.0, 1, Some(snap(0.94, 0.06, 30.0, 1))),
    ];

    let idx_uncertain = run_policy(&policy, &edges_uncertain, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx_uncertain, 3);

    // Case 2: most-visited baseline is winning => minimize => short wins.
    let edges_winning = [
        edge(0, &b0, 0.0, 10, Some(snap(0.39, 0.61, 20.0, 1))),
        edge(1, &b1, 0.0, 11, Some(snap(0.90, 0.10, 20.0, 1))),
        edge(2, &c0, 0.0, 1, Some(snap(0.93, 0.07, 5.0, 1))),
        edge(3, &c1, 0.0, 1, Some(snap(0.94, 0.06, 30.0, 1))),
    ];

    let idx_winning = run_policy(&policy, &edges_winning, 100, &TestState { ptm: 1 }, 1);
    assert_eq!(idx_winning, 2);
}

#[test]
fn policy_matches_reference_two_pass_for_multiple_scenarios() {
    let options_policy = default_options();
    let options_ref = default_options();
    let policy = TestPolicy::new(
        ConstantCpuct(0.25),
        options_policy,
        NoTrajectoryTerminal::default(),
    );

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;
    let a3 = 3u8;

    let scenarios: Vec<Scenario<'_>> = vec![
        (
            vec![
                edge(0, &a0, 0.6, 10, Some(snap(0.9, 0.1, 20.0, 1))),
                edge(1, &a1, 0.2, 1, Some(snap(0.5, 0.5, 10.0, 1))),
                edge(2, &a2, 0.2, 1, Some(snap(0.5, 0.5, 30.0, 1))),
            ],
            100,
            TestState { ptm: 1 },
            1,
        ),
        (
            vec![
                edge(0, &a0, 0.6, 10, Some(snap(0.1, 0.9, 20.0, 1))),
                edge(1, &a1, 0.2, 1, Some(snap(0.5, 0.5, 10.0, 1))),
                edge(2, &a2, 0.2, 1, Some(snap(0.5, 0.5, 30.0, 1))),
            ],
            100,
            TestState { ptm: 1 },
            1,
        ),
        (
            vec![
                edge(0, &a0, 0.5, 10, Some(snap(0.6, 0.4, 20.0, 1))),
                edge(1, &a1, 0.25, 1, Some(snap(0.40, 0.60, 10.0, 1))),
                edge(2, &a2, 0.25, 1, Some(snap(0.41, 0.59, 30.0, 1))),
                edge(3, &a3, 0.0, 0, None),
            ],
            25,
            TestState { ptm: 1 },
            0,
        ),
    ];

    for (edges, node_visits, state, depth) in scenarios {
        let expected =
            reference_two_pass_select(&edges, node_visits, &state, depth, 0.25, &options_ref);
        let actual = run_policy(&policy, &edges, node_visits, &state, depth);
        assert_eq!(actual, expected);
    }
}

#[test]
fn map_moves_left_to_one_hot_basic_cases() {
    assert_eq!(map_moves_left_to_one_hot(10.0, 0), Vec::<f32>::new());

    assert_eq!(map_moves_left_to_one_hot(0.0, 4), vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(map_moves_left_to_one_hot(1.0, 4), vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(map_moves_left_to_one_hot(2.0, 4), vec![0.0, 1.0, 0.0, 0.0]);

    // Rounds.
    assert_eq!(map_moves_left_to_one_hot(2.49, 4), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(map_moves_left_to_one_hot(2.50, 4), vec![0.0, 0.0, 1.0, 0.0]);

    // Clamps to size.
    assert_eq!(
        map_moves_left_to_one_hot(999.0, 4),
        vec![0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
#[should_panic]
fn map_moves_left_to_one_hot_panics_on_nan() {
    let _ = map_moves_left_to_one_hot(f32::NAN, 4);
}

#[test]
#[should_panic]
fn map_moves_left_to_one_hot_panics_on_negative() {
    let _ = map_moves_left_to_one_hot(-1.0, 4);
}

#[test]
#[should_panic]
fn map_moves_left_to_one_hot_panics_on_infinity() {
    let _ = map_moves_left_to_one_hot(f32::INFINITY, 4);
}

#[test]
fn moves_left_expected_value_is_weighted_index_sum() {
    let ev = moves_left_expected_value([1.0, 0.0, 0.0].into_iter());
    assert!((ev - 1.0).abs() < 1e-6);

    let ev = moves_left_expected_value([0.0, 1.0, 0.0].into_iter());
    assert!((ev - 2.0).abs() < 1e-6);

    let ev = moves_left_expected_value([0.5, 0.5].into_iter());
    assert!((ev - 1.5).abs() < 1e-6);
}

#[test]
fn moves_left_expected_value_empty_iterator_returns_zero() {
    let ev = moves_left_expected_value(std::iter::empty());
    assert_eq!(ev, 0.0);
}

// ---------------------------------------------------------------------------
// MovesLeftSnapshot unit tests
// ---------------------------------------------------------------------------

#[test]
fn snapshot_zero_returns_zero_for_all_accessors() {
    let s = MovesLeftSnapshot::default();
    assert_eq!(s.player_value(1), 0.0);
    assert_eq!(s.player_value(2), 0.0);
    assert_eq!(s.game_length(), 0.0);
}

#[test]
#[should_panic]
fn snapshot_player_value_panics_on_invalid_player_index() {
    let s = snap(0.5, 0.5, 10.0, 1);
    let _ = s.player_value(0);
}

#[test]
fn snapshot_display_includes_player_and_game_length_fields() {
    let s = snap(0.35, 0.65, 20.0, 1);
    let display = format!("{}", s);
    assert!(display.contains("p1:"));
    assert!(display.contains("p2:"));
    assert!(display.contains("gl:"));
}

// ---------------------------------------------------------------------------
// MovesLeftValueModel tests
// ---------------------------------------------------------------------------

struct StubPredictions {
    p1: f32,
    p2: f32,
    gl: f32,
}

impl PlayerValue for StubPredictions {
    fn player_value(&self, player: usize) -> f32 {
        match player {
            1 => self.p1,
            2 => self.p2,
            _ => panic!(),
        }
    }
}

impl GameLength for StubPredictions {
    fn game_length(&self) -> f32 {
        self.gl
    }
}

struct StubTerminal {
    p1: f32,
    p2: f32,
}

impl PlayerValue for StubTerminal {
    fn player_value(&self, player: usize) -> f32 {
        match player {
            1 => self.p1,
            2 => self.p2,
            _ => panic!(),
        }
    }
}

#[test]
fn value_model_pred_snapshot_stores_player_values_and_game_length() {
    let model = MovesLeftValueModel::<(), StubPredictions, StubTerminal>::new();
    let preds = StubPredictions {
        p1: 0.6,
        p2: 0.4,
        gl: 12.0,
    };

    let snap = model.pred_snapshot(&(), &preds);

    assert_eq!(snap.total_weight, 1);
    assert!((snap.player_value(1) - 0.6).abs() < 1e-6);
    assert!((snap.player_value(2) - 0.4).abs() < 1e-6);
    assert!((snap.game_length() - 12.0).abs() < 1e-6);
}

#[test]
fn value_model_terminal_snapshot_sets_game_length_to_zero() {
    let model = MovesLeftValueModel::<(), StubPredictions, StubTerminal>::new();
    let terminal = StubTerminal { p1: 0.0, p2: 1.0 };

    let snap = model.terminal_snapshot(&(), &terminal);

    assert_eq!(snap.total_weight, 1);
    assert!((snap.player_value(1) - 0.0).abs() < 1e-6);
    assert!((snap.player_value(2) - 1.0).abs() < 1e-6);
    // Terminal positions have no moves remaining; game_length_sum is stored as 0.
    assert_eq!(snap.game_length(), 0.0);
}

// ---------------------------------------------------------------------------
// Player-2 perspective selection-policy tests
// ---------------------------------------------------------------------------

#[test]
fn player_two_winning_baseline_prefers_shorter_game_length() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 1.0,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;

    // Baseline edge 0: most visited, p2=0.9 (winning for player 2), len=20.
    // Candidates 1 and 2 have equal Q but shorter vs longer game; minimize should pick edge 1.
    let edges = [
        edge(0, &a0, 0.0, 10, Some(snap(0.1, 0.9, 20.0, 1))),
        edge(1, &a1, 0.0, 1, Some(snap(0.5, 0.5, 10.0, 1))),
        edge(2, &a2, 0.0, 1, Some(snap(0.5, 0.5, 30.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 2 }, 1);
    assert_eq!(idx, 1);
}

#[test]
fn player_two_losing_baseline_prefers_longer_game_length() {
    let options = MovesLeftStrategyOptions {
        moves_left_threshold: 0.7,
        moves_left_scale: 20.0,
        moves_left_factor: 1.0,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;
    let a2 = 2u8;

    // Baseline edge 0: most visited, p2=0.1 (losing for player 2), len=20.
    // When losing, maximize game length; candidate 2 (len=30) should win.
    let edges = [
        edge(0, &a0, 0.0, 10, Some(snap(0.9, 0.1, 20.0, 1))),
        edge(1, &a1, 0.0, 1, Some(snap(0.5, 0.5, 10.0, 1))),
        edge(2, &a2, 0.0, 1, Some(snap(0.5, 0.5, 30.0, 1))),
    ];

    let idx = run_policy(&policy, &edges, 100, &TestState { ptm: 2 }, 1);
    assert_eq!(idx, 2);
}

#[test]
fn player_two_fpu_root_vs_non_root() {
    let options = MovesLeftStrategyOptions {
        fpu_root: 0.9,
        fpu: 0.1,
        ..default_options()
    };
    let policy = TestPolicy::new(ConstantCpuct(0.0), options, NoTrajectoryTerminal::default());

    let a0 = 0u8;
    let a1 = 1u8;

    // Edge 0 has a snapshot with p2=0.5; edge 1 has no snapshot (uses fpu).
    let edges = [
        edge(0, &a0, 0.0, 0, Some(snap(0.5, 0.5, 10.0, 1))),
        edge(1, &a1, 0.0, 0, None),
    ];

    // Root: fpu_root=0.9 > p2=0.5, so edge 1 wins.
    let root_idx = run_policy(&policy, &edges, 1, &TestState { ptm: 2 }, 0);
    assert_eq!(root_idx, 1);

    // Non-root: fpu=0.1 < p2=0.5, so edge 0 wins.
    let child_idx = run_policy(&policy, &edges, 1, &TestState { ptm: 2 }, 1);
    assert_eq!(child_idx, 0);
}
