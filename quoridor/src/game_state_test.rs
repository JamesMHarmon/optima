#[cfg(test)]
mod tests {
    use super::super::{Action, GameState, Value};
    use engine::game_state::GameState as GameStateTrait;

    fn intersects(actions: &[Action], exclusions: &[Action]) -> bool {
        actions.iter().any(|a| exclusions.iter().any(|a2| a == a2))
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p1() {
        let game_state = GameState::initial();
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f1".parse::<Action>().unwrap(),
                "d1".parse::<Action>().unwrap(),
                "e2".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"f1".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "e8".parse::<Action>().unwrap(),
                "f9".parse::<Action>().unwrap(),
                "d9".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"d2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2v".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(valid_actions, vec!("e2".parse::<Action>().unwrap()));
    }

    #[test]
    fn test_get_valid_pawn_move_actions_vertical_wall_top() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"d2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2v".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "e1".parse::<Action>().unwrap(),
                "e3".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_horizontal_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"d9h".parse::<Action>().unwrap());
        game_state.take_action(&"e2h".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f1".parse::<Action>().unwrap(),
                "d1".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"f1".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f9".parse::<Action>().unwrap(),
                "d9".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"f9".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "g1".parse::<Action>().unwrap(),
                "e1".parse::<Action>().unwrap()
            )
        );
    }

    #[test]
    fn test_get_valid_pawn_move_actions_blocked() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());

        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();
        assert_eq!(
            valid_actions,
            vec!(
                "e4".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap(),
                "e7".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"e5h".parse::<Action>().unwrap());
        game_state.take_action(&"a2h".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f5".parse::<Action>().unwrap(),
                "d5".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap(),
                "e7".parse::<Action>().unwrap()
            )
        );

        game_state.take_action(&"e7h".parse::<Action>().unwrap());
        let valid_actions = game_state.valid_pawn_move_actions().collect::<Vec<_>>();

        assert_eq!(
            valid_actions,
            vec!(
                "f5".parse::<Action>().unwrap(),
                "d5".parse::<Action>().unwrap(),
                "f6".parse::<Action>().unwrap(),
                "d6".parse::<Action>().unwrap()
            )
        );
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
        game_state.take_action(&"d2h".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c2h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_on_vertical_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e6v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec!["e6h".parse::<Action>().unwrap()];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
            "d2h".parse::<Action>().unwrap(),
            "d3h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_other_player() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
            "d3h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_vert_horz() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"c2v".parse::<Action>().unwrap());
        game_state.take_action(&"e2v".parse::<Action>().unwrap());
        game_state.take_action(&"c3h".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "c2h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
            "d3h".parse::<Action>().unwrap(),
            "b3h".parse::<Action>().unwrap(),
            "c3h".parse::<Action>().unwrap(),
            "d3h".parse::<Action>().unwrap(),
            "e3h".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_horizontal_wall_actions_blocking_path_edge() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2v".parse::<Action>().unwrap());
        game_state.take_action(&"e3h".parse::<Action>().unwrap());
        game_state.take_action(&"c3h".parse::<Action>().unwrap());
        game_state.take_action(&"b4v".parse::<Action>().unwrap());

        let valid_actions = game_state
            .valid_horizontal_wall_actions()
            .collect::<Vec<_>>();
        let excludes_actions = vec![
            "a3h".parse::<Action>().unwrap(),
            "a4h".parse::<Action>().unwrap(),
            "a5h".parse::<Action>().unwrap(),
            "b3h".parse::<Action>().unwrap(),
            "b4h".parse::<Action>().unwrap(),
            "c3h".parse::<Action>().unwrap(),
            "d3h".parse::<Action>().unwrap(),
            "e3h".parse::<Action>().unwrap(),
            "e2h".parse::<Action>().unwrap(),
            "f3h".parse::<Action>().unwrap(),
        ];
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
        game_state.take_action(&"e6v".parse::<Action>().unwrap());

        let valid_actions = game_state.valid_vertical_wall_actions().collect::<Vec<_>>();
        let excludes_actions = vec![
            "e5v".parse::<Action>().unwrap(),
            "e6v".parse::<Action>().unwrap(),
            "e7v".parse::<Action>().unwrap(),
        ];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_vertical_wall_actions_on_horizontal_wall() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e6h".parse::<Action>().unwrap());

        let valid_actions = game_state.valid_vertical_wall_actions().collect::<Vec<_>>();
        let excludes_actions = vec!["e6v".parse::<Action>().unwrap()];
        let intersects = intersects(&valid_actions, &excludes_actions);

        assert!(!intersects);
        assert_eq!(valid_actions.len(), 64 - excludes_actions.len());
    }

    #[test]
    fn test_get_valid_wall_actions_on_all_walls_placed() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"a2h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"c2h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"e2h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"g2h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"a3h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"c3h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"e3h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());
        game_state.take_action(&"g3h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());
        game_state.take_action(&"a4h".parse::<Action>().unwrap());
        game_state.take_action(&"f9".parse::<Action>().unwrap());

        // 9 walls placed
        let valid_actions = game_state.valid_horizontal_wall_actions();
        assert_eq!(valid_actions.count(), 46);

        game_state.take_action(&"c4h".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        // 10 walls placed so we shouldn't be able to place anymore, horizontal or vertical
        let valid_horizontal_actions = game_state.valid_horizontal_wall_actions();
        assert_eq!(valid_horizontal_actions.count(), 0);

        let valid_vertical_actions = game_state.valid_vertical_wall_actions();
        assert_eq!(valid_vertical_actions.count(), 0);
    }

    #[test]
    fn test_is_terminal_p2() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        game_state.take_action(&"e1".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([0.0, 1.0])));
    }

    #[test]
    fn test_is_terminal_p1() {
        let mut game_state = GameState::initial();
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e5".parse::<Action>().unwrap());
        game_state.take_action(&"e4".parse::<Action>().unwrap());
        game_state.take_action(&"e6".parse::<Action>().unwrap());
        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e7".parse::<Action>().unwrap());
        game_state.take_action(&"e2".parse::<Action>().unwrap());
        game_state.take_action(&"e8".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, None);

        game_state.take_action(&"e3".parse::<Action>().unwrap());
        game_state.take_action(&"e9".parse::<Action>().unwrap());

        let is_terminal = game_state.is_terminal();
        assert_eq!(is_terminal, Some(Value([1.0, 0.0])));
    }

    #[test]
    fn test_move_number() {
        let mut game_state = GameState::initial();
        assert_eq!(game_state.move_number(), 1);

        game_state.take_action(&"e2".parse::<Action>().unwrap());
        assert_eq!(game_state.move_number(), 1);

        game_state.take_action(&"e9".parse::<Action>().unwrap());
        assert_eq!(game_state.move_number(), 2);

        game_state.take_action(&"b5h".parse::<Action>().unwrap());
        assert_eq!(game_state.move_number(), 2);

        game_state.take_action(&"d5v".parse::<Action>().unwrap());
        assert_eq!(game_state.move_number(), 3);
    }
}
