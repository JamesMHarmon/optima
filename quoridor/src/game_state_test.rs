#[cfg(test)]
mod tests {
    use super::super::{Action, GameState, Value};
    use engine::game_state::GameState as GameStateTrait;

    macro_rules! take_actions {
        ($game_state:expr, [$($action:expr),*]) => {
            $(
                $game_state.take_action(&$action.parse::<Action>().unwrap());
            )*
        };
    }

    macro_rules! actions {
        ($($action:expr),*) => {
            vec![$($action.parse::<Action>().unwrap()),*]
        };
    }

    fn intersects(actions: &[Action], exclusions: &[Action]) -> bool {
        actions.iter().any(|a| exclusions.iter().any(|a2| a == a2))
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p1() {
        let game_state = GameState::initial();
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["f1", "d1", "e2"]);
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p2() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["f1"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["e8", "f9", "d9"]);
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["d2v", "e2v"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["e2"]);
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall_top() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e2", "e7", "d2v", "e2v"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["e1", "e3"]);
    }

    #[test]
    fn test_get_valid_pawn_move_actions_horizontal_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["d9h", "e2h"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["f1", "d1"]);

        take_actions![game_state, ["f1"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["f9", "d9"]);

        take_actions![game_state, ["f9"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["g1", "e1"]);
    }

    #[test]
    fn test_get_valid_pawn_move_actions_blocked() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e2", "e8", "e3", "e7", "e4", "e6", "e5"]];

        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();
        assert_eq!(valid_actions, actions!["e4", "f6", "d6", "e7"]);

        take_actions![game_state, ["e5h", "a2h"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["f5", "d5", "f6", "d6", "e7"]);

        take_actions![game_state, ["e7h"]];
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, actions!["f5", "d5", "f6", "d6"]);
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();

        let cols = || 'a'..='h';
        let rows = 2..=9;

        let mut actions = Vec::new();

        for row in rows {
            for col in cols().rev() {
                actions.push(format!("{}{}h", col, row).parse().unwrap());
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_horizontal_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["d2h"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = actions!["c2h", "d2h", "e2h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_vertical_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e6v"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = actions!["e6h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["c2v", "e2v"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = actions!["c2h", "e2h", "d2h", "d3h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_other_player() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["c2v", "e2v", "e2"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = actions!["c2h", "e2h", "d3h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_vert_horz() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["c2v", "e2v", "c3h"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = actions!["c2h", "e2h", "d3h", "b3h", "c3h", "d3h", "e3h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_edge() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e2v", "e3h", "c3h", "b4v"]];

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions =
            actions!["a3h", "a4h", "a5h", "b3h", "b4h", "c3h", "d3h", "e3h", "e2h", "f3h"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_initial() {
        let game_state = GameState::initial();
        let valid_actions = game_state.valid_vertical_wall_actions().collect::<Vec<_>>();

        let cols = || 'a'..='h';
        let rows = 2..=9;

        let mut actions = Vec::new();

        for row in rows {
            for col in cols().rev() {
                actions.push(format!("{}{}v", col, row).parse().unwrap());
            }
        }

        assert_eq!(valid_actions.len(), actions.len());
        assert_eq!(valid_actions, actions);
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_vertical_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e6v"]];

        let valid_actions = game_state.valid_vertical_wall_actions().collect::<Vec<_>>();
        let excludes_actions = actions!["e5v", "e6v", "e7v"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_horizontal_wall() {
        let mut game_state = GameState::initial();
        take_actions![game_state, ["e6h"]];

        let valid_actions = game_state.valid_vertical_wall_actions().collect::<Vec<_>>();
        let excludes_actions = actions!["e6v"];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_wall_actions_on_all_walls_placed() {
        let mut game_state = GameState::initial();
        take_actions![
            game_state,
            [
                "a2h", "f9", "c2h", "e9", "e2h", "f9", "g2h", "e9", "a3h", "f9", "c3h", "e9",
                "e3h", "f9", "g3h", "e9", "a4h", "f9"
            ]
        ];

        // 9 walls placed
        let valid_actions = game_state.valid_horizontal_wall_actions();
        assert_eq!(valid_actions.count(), 46);

        take_actions![game_state, ["c4h", "e9"]];

        // 10 walls placed so we shouldn't be able to place anymore, horizontal or vertical
        let valid_horizontal_actions = game_state.valid_horizontal_wall_actions();
        assert_eq!(valid_horizontal_actions.count(), 0);

        let valid_vertical_actions = game_state.valid_vertical_wall_actions();
        assert_eq!(valid_vertical_actions.count(), 0);
    }

    #[test]
    fn test_is_terminal_p2() {
        let mut game_state = GameState::initial();
        take_actions![
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8"]
        ];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["e1"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["p"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1() {
        let mut game_state = GameState::initial();
        take_actions![
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8"]
        ];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["e3", "e9"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["p"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([1.0, 0.0])));
        assert_eq!(game_state.victory_margin(), 2);
    }

    #[test]
    fn test_move_number() {
        let mut game_state = GameState::initial();
        assert_eq!(game_state.move_number(), 1);

        take_actions![game_state, ["e2"]];
        assert_eq!(game_state.move_number(), 1);

        take_actions![game_state, ["e9"]];
        assert_eq!(game_state.move_number(), 2);

        take_actions![game_state, ["b5h"]];
        assert_eq!(game_state.move_number(), 2);

        take_actions![game_state, ["d5v"]];
        assert_eq!(game_state.move_number(), 3);
    }

    #[test]
    fn test_scoring_phase_excludes_moves() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8"]
        );

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["e3", "e9"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["a2v", "e2"]];

        let valid_pawn_move_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();
        let excludes_actions = actions!["e8", "d9", "f9"];

        let intersects = intersects(&valid_pawn_move_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_pawn_move_actions.len(), 0);

        let valid_vertical_wall_actions =
            game_state.valid_vertical_wall_actions().collect::<Vec<_>>();

        assert_eq!(valid_vertical_wall_actions.len(), (8 * 8) - 2);

        let valid_horizontal_wall_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();

        assert_eq!(valid_horizontal_wall_actions.len(), (8 * 8) - 1);

        take_actions![game_state, ["p"]];
        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));

        assert_eq!(game_state.victory_margin(), 1);
    }

    #[test]
    fn test_scoring_phase_excludes_moves_p2() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8"]
        );

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        take_actions![game_state, ["e1", "a2v"]];

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        let valid_pawn_move_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();
        let includes_actions = actions!["f8", "d8", "e7", "e9"];

        let has_intersects = intersects(&valid_pawn_move_actions, &includes_actions);

        assert!(has_intersects);

        take_actions![game_state, ["e7"]];

        let valid_pawn_move_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();
        let excludes_actions = actions!["f1", "d1", "e2"];
        
        let has_intersects = intersects(&valid_pawn_move_actions, &excludes_actions);

        assert!(!has_intersects);
        assert_eq!(valid_pawn_move_actions.len(), 0);

        let valid_vertical_wall_actions =
            game_state.valid_vertical_wall_actions().collect::<Vec<_>>();

        assert_eq!(valid_vertical_wall_actions.len(), (8 * 8) - 2);

        let valid_horizontal_wall_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();

        assert_eq!(valid_horizontal_wall_actions.len(), (8 * 8) - 1);

        take_actions![game_state, ["p"]];
        assert_eq!(game_state.is_terminal(), Some(Value([0.0, 1.0])));

        assert_eq!(game_state.victory_margin(), 2);
    }

    #[test]
    fn test_turn_is_repeated_for_goaling_player_p1() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8", "e3"]
        );

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e9"]];

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");
    }

    #[test]
    fn test_turn_passes_for_passing_player_p1() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8", "e3"]
        );

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e9"]];

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["p"]];

        assert!(!game_state.p1_turn_to_move());
        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
        assert_eq!(game_state.victory_margin(), 2);
    }

    #[test]
    fn test_turn_passes_for_placing_wall_p1() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8", "e3"]
        );

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e9"]];

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["a2v"]];

        assert!(!game_state.p1_turn_to_move());
        assert_eq!(game_state.is_terminal(), None);
    }

    #[test]
    fn test_turn_passes_for_placing_last_wall_p1() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8", "e3"]
        );

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e9"]];

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["a1v", "e2", "a2v", "e3", "a3v", "e2", "a4v", "e3", "a5v", "e2", "a6v", "e3", "a7v", "e2", "a8v", "e3", "c2v", "e2", "c3v"]];

        assert!(!game_state.p1_turn_to_move());
        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
        assert_eq!(game_state.victory_margin(), 1);
    }

    #[test]
    fn test_both_players_goal() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8", "f2"]
        );

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e9"]];

        assert!(game_state.p1_turn_to_move(), "p1 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["a2v"]];

        assert!(!game_state.p1_turn_to_move());

        take_actions![game_state, ["f1"]];

        assert_eq!(game_state.is_terminal(), Some(Value([1.0, 0.0])));
        assert_eq!(game_state.victory_margin(), 0);
    }

    #[test]
    fn test_turn_is_repeated_for_goaling_player_p2() {
        let mut game_state = GameState::initial();
        take_actions!(
            game_state,
            ["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "e6", "e3", "e7", "e2", "e8"]
        );

        assert!(!game_state.p1_turn_to_move(), "p2 should get a consecutive turn after reaching the goal.");

        take_actions![game_state, ["e1"]];

        assert!(!game_state.p1_turn_to_move(), "p2 should get a consecutive turn after reaching the goal.");
    }
}
