use super::*;

type Scenario<'a> = (
    TestState,
    u32,
    u32,
    Vec<EdgeInfo<'a, u8, VictoryMarginSnapshot>>,
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

fn snap(p1: f32, p2: f32, vm: f32, weight: u32) -> VictoryMarginSnapshot {
    VictoryMarginSnapshot {
        p1_sum: (p1 as f64) * (weight as f64),
        p2_sum: (p2 as f64) * (weight as f64),
        victory_margin_sum: (vm as f64) * (weight as f64),
        total_weight: weight,
    }
}

fn edge<'a, A>(
    edge_index: usize,
    action: &'a A,
    policy_prior: f32,
    visits: u32,
    snapshot: Option<VictoryMarginSnapshot>,
) -> EdgeInfo<'a, A, VictoryMarginSnapshot> {
    EdgeInfo {
        edge_index,
        action,
        policy_prior,
        visits,
        snapshot,
    }
}

fn run_policy<'a, A>(
    policy: &VictoryMarginSelectionPolicy<ConstantCpuct, TestState>,
    edges: &'a [EdgeInfo<'a, A, VictoryMarginSnapshot>],
    node_visits: u32,
    state: &TestState,
    depth: u32,
) -> usize
where
    A: 'a,
{
    policy.select_edge(
        edges.iter().map(|e| EdgeInfo {
            edge_index: e.edge_index,
            action: e.action,
            policy_prior: e.policy_prior,
            visits: e.visits,
            snapshot: e.snapshot,
        }),
        node_visits,
        state,
        depth,
    )
}

fn reference_two_pass_select<'a, A>(
    edges: &[EdgeInfo<'a, A, VictoryMarginSnapshot>],
    node_visits: u32,
    state: &TestState,
    depth: u32,
    cpuct: f32,
    options: &VictoryMarginStrategyOptions,
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
    let root_sqrt = (node_visits as f32).sqrt();
    let player_index = state.player_to_move();

    let mut baseline_visits = 0u32;
    let mut baseline_snap: Option<VictoryMarginSnapshot> = None;
    for e in edges {
        if e.visits == 0 {
            continue;
        }
        let Some(s) = e.snapshot else {
            continue;
        };
        if e.visits > baseline_visits {
            baseline_visits = e.visits;
            baseline_snap = Some(s);
        }
    }

    let directive =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::directive_from_baseline(
            baseline_snap,
            options.victory_margin_threshold,
            player_index,
        );

    let mut best_index = 0usize;
    let mut best_score = f32::MIN;

    for e in edges {
        let nsa = e.visits;
        let psa = e.policy_prior;
        let usa = cpuct * psa * root_sqrt / (1.0 + nsa as f32);
        let qsa = e
            .snapshot
            .map(|s| s.value_for_player(player_index))
            .unwrap_or(fpu);

        let vm_adj = if nsa == 0 {
            0.0
        } else {
            e.snapshot
                .map(|s| s.victory_margin() * options.victory_margin_factor)
                .unwrap_or(0.0)
        };

        let score = match directive {
            VictoryMarginDirective::None => qsa + usa,
            VictoryMarginDirective::MaximizeVictoryMargin => qsa + usa + vm_adj,
            VictoryMarginDirective::MinimizeVictoryMargin => qsa + usa - vm_adj,
        };

        if score > best_score {
            best_score = score;
            best_index = e.edge_index;
        }
    }

    best_index
}

#[test]
fn select_matches_reference_multiple_scenarios() {
    static ACTIONS: [u8; 6] = [1, 2, 3, 4, 5, 6];
    let cpuct = 1.0;
    let policy = VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(
        ConstantCpuct(cpuct),
        VictoryMarginStrategyOptions {
            fpu: 0.25,
            fpu_root: 0.30,
            victory_margin_threshold: 0.75,
            victory_margin_factor: 0.10,
        },
    );

    let scenarios: Vec<Scenario<'_>> = vec![
        (
            TestState { ptm: 1 },
            0,
            100,
            vec![
                edge(0, &ACTIONS[0], 0.4, 0, None),
                edge(1, &ACTIONS[1], 0.3, 10, Some(snap(0.9, 0.1, 0.5, 10))),
                edge(2, &ACTIONS[2], 0.2, 25, Some(snap(0.8, 0.2, 0.9, 25))),
                edge(3, &ACTIONS[3], 0.1, 50, Some(snap(0.6, 0.4, 0.1, 50))),
            ],
        ),
        (
            TestState { ptm: 2 },
            0,
            80,
            vec![
                edge(0, &ACTIONS[0], 0.5, 0, None),
                edge(1, &ACTIONS[1], 0.2, 12, Some(snap(0.8, 0.2, 0.8, 12))),
                edge(2, &ACTIONS[2], 0.2, 40, Some(snap(0.9, 0.1, 0.6, 40))),
                edge(3, &ACTIONS[3], 0.1, 41, Some(snap(0.2, 0.8, 0.3, 41))),
            ],
        ),
        (
            TestState { ptm: 1 },
            2,
            1,
            vec![
                edge(0, &ACTIONS[0], 0.6, 0, None),
                edge(1, &ACTIONS[1], 0.4, 0, Some(snap(0.2, 0.8, 0.9, 1))),
            ],
        ),
    ];

    for (state, depth, node_visits, edges) in scenarios {
        let selected = run_policy(&policy, &edges, node_visits, &state, depth);
        let expected =
            reference_two_pass_select(&edges, node_visits, &state, depth, cpuct, &policy.options);
        assert_eq!(selected, expected);
    }
}

#[test]
fn directive_threshold_ge_one_is_none() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        victory_margin_threshold: 1.0,
        victory_margin_factor: 1.0,
    };

    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 2] = [1, 2];
    let edges = vec![
        edge(0, &A[0], 0.5, 10, Some(snap(0.99, 0.01, 0.9, 10))),
        edge(1, &A[1], 0.5, 10, Some(snap(0.80, 0.20, 0.1, 10))),
    ];

    let selected = run_policy(&policy, &edges, 100, &state, 0);
    // With cpuct=0 and fpu=0, selection under None becomes max Q; edge 0 has higher Q.
    assert_eq!(selected, 0);
}

#[test]
fn no_baseline_means_no_directive_and_picks_best_base() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.4,
        fpu_root: 0.4,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 10.0,
    };

    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 3] = [1, 2, 3];
    let edges = vec![
        edge(0, &A[0], 0.0, 0, None),
        edge(1, &A[1], 0.0, 0, None),
        edge(2, &A[2], 0.0, 0, None),
    ];

    // No visited edges => baseline is None => directive None => VM ignored.
    // With cpuct=0, it picks best Q; for unvisited edges Q=fpu=0.4, for visited it would use snapshot,
    // but all visits are 0 so snapshot is ignored and Q is also fpu.
    let selected = run_policy(&policy, &edges, 10, &state, 0);
    assert_eq!(selected, 0);
}

#[test]
fn winning_baseline_maximizes_victory_margin() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 1.0,
    };
    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 3] = [1, 2, 3];
    // Baseline is edge 0 (most visited), and it's winning.
    // Among edges 1 and 2, Q is identical but VM differs; maximize should choose edge 2.
    let edges = vec![
        edge(0, &A[0], 0.0, 100, Some(snap(0.9, 0.1, 0.1, 100))),
        edge(1, &A[1], 0.0, 10, Some(snap(0.9, 0.1, 0.2, 10))),
        edge(2, &A[2], 0.0, 10, Some(snap(0.9, 0.1, 0.8, 10))),
    ];

    let selected = run_policy(&policy, &edges, 100, &state, 0);
    assert_eq!(selected, 2);
}

#[test]
fn losing_baseline_minimizes_victory_margin() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 1.0,
    };
    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 3] = [1, 2, 3];
    // Baseline is edge 0 (most visited), and it's losing.
    // Among edges 1 and 2, Q is identical but VM differs; minimize should choose edge 1.
    let edges = vec![
        edge(0, &A[0], 0.0, 100, Some(snap(0.1, 0.9, 0.5, 100))),
        edge(1, &A[1], 0.0, 10, Some(snap(0.1, 0.9, 0.1, 10))),
        edge(2, &A[2], 0.0, 10, Some(snap(0.1, 0.9, 0.9, 10))),
    ];

    let selected = run_policy(&policy, &edges, 100, &state, 0);
    assert_eq!(selected, 1);
}

#[test]
fn uncertain_baseline_does_not_bias_victory_margin() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 1.0,
    };
    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 3] = [1, 2, 3];
    // Baseline (edge 0) is in the middle => directive None.
    // Edge 2 has huge VM but lower Q; with no bias we should pick edge 1.
    let edges = vec![
        edge(0, &A[0], 0.0, 100, Some(snap(0.6, 0.4, 0.5, 100))),
        edge(1, &A[1], 0.0, 10, Some(snap(0.7, 0.3, 0.0, 10))),
        edge(2, &A[2], 0.0, 10, Some(snap(0.69, 0.31, 100.0, 10))),
    ];

    let selected = run_policy(&policy, &edges, 100, &state, 0);
    assert_eq!(selected, 1);
}

#[test]
fn unvisited_edges_do_not_get_victory_margin_bonus() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.1,
        fpu_root: 0.1,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 10.0,
    };

    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 2] = [1, 2];
    // Baseline edge 1 makes us "winning" so directive maximize.
    // Edge 0 is unvisited but has a snapshot with enormous VM; it should not benefit.
    let edges = vec![
        edge(0, &A[0], 0.0, 0, Some(snap(0.5, 0.5, 999.0, 1))),
        edge(1, &A[1], 0.0, 10, Some(snap(0.9, 0.1, 0.1, 10))),
    ];

    let selected = run_policy(&policy, &edges, 10, &state, 0);
    assert_eq!(selected, 1);
}

#[test]
fn baseline_is_most_visited_not_best_q() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.0,
        fpu_root: 0.0,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 1.0,
    };

    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 3] = [1, 2, 3];
    // Edge 0: most visited but losing => directive minimize.
    // Edge 1: fewer visits but winning.
    // Between edges 1 and 2 with same Q, minimize prefers smaller VM.
    let edges = vec![
        edge(0, &A[0], 0.0, 100, Some(snap(0.1, 0.9, 0.5, 100))),
        edge(1, &A[1], 0.0, 10, Some(snap(0.9, 0.1, 0.9, 10))),
        edge(2, &A[2], 0.0, 10, Some(snap(0.9, 0.1, 0.1, 10))),
    ];

    let selected = run_policy(&policy, &edges, 100, &state, 0);
    assert_eq!(selected, 2);
}

#[test]
fn fpu_root_used_at_depth_zero() {
    let state = TestState { ptm: 1 };
    let options = VictoryMarginStrategyOptions {
        fpu: 0.2,
        fpu_root: 0.8,
        victory_margin_threshold: 0.75,
        victory_margin_factor: 0.0,
    };

    let policy =
        VictoryMarginSelectionPolicy::<ConstantCpuct, TestState>::new(ConstantCpuct(0.0), options);

    static A: [u8; 2] = [1, 2];
    let edges = vec![edge(0, &A[0], 0.0, 0, None), edge(1, &A[1], 0.0, 0, None)];

    // With everything identical, selection is deterministic (first edge), but we can at least
    // validate the score path doesn't panic and matches the reference implementation.
    let selected = run_policy(&policy, &edges, 10, &state, 0);
    let expected = reference_two_pass_select(&edges, 10, &state, 0, 0.0, &policy.options);
    assert_eq!(selected, expected);

    let selected_non_root = run_policy(&policy, &edges, 10, &state, 1);
    let expected_non_root = reference_two_pass_select(&edges, 10, &state, 1, 0.0, &policy.options);
    assert_eq!(selected_non_root, expected_non_root);
}
