use std::cmp::Ordering;
use std::fmt::{self, Debug, Display, Formatter};

#[allow(non_snake_case)]
pub struct NodeDetails<A, P, PV> {
    pub visits: usize,
    pub predictions: P,
    pub children: Vec<EdgeDetails<A, PV>>,
}

impl<A: Display, P, PV> Display for NodeDetails<A, P, PV> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let actions = format!(
            "[{}]",
            self.children
                .iter()
                .fold(String::new(), |acc, (a, puct)| acc
                    + &format!("\n\t(A: {}, {}),", a, puct))
        );

        write!(
            f,
            "V: {visits}, Actions: {actions}",
            visits = self.visits,
            actions = actions
        )
    }
}

impl<A: Debug + Display, P, PV> Debug for NodeDetails<A, P, PV> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(PartialEq)]
#[allow(non_snake_case)]
pub struct EdgeDetails<A, PV> {
    pub action: A,
    pub Nsa: usize,
    pub Qsa: f32,
    pub Psa: f32,
    pub Usa: f32,
    /// Expected game length calculated by each child node's moves left score normalized to expected game length.
    pub Msa: f32,
    pub M: f32,
    /// Neural net output of the number of moves left in the game.
    pub moves_left_score: f32,
    pub game_length: f32,
    pub cpuct: f32,
    pub puct_score: f32,
}

impl<A, PV> Display for EdgeDetails<A, PV> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Nsa: {Nsa}, Qsa: {Qsa:.3}, Msa: {Msa:.2}, Psa: {Psa:.3}, Usa: {Usa:.2}, cpuct: {cpuct:.2}, avg_game_length: {game_length:.1} moves_left_head_score: {moves_left_score:.1}, PUCT: {PUCT:.3}",
            Nsa = self.Nsa,
            Qsa = self.Qsa,
            Psa = self.Psa,
            Usa = self.Usa,
            Msa = self.Msa,
            moves_left_score = self.moves_left_score,
            game_length = self.game_length,
            cpuct = self.cpuct,
            PUCT = self.PUCT,
        )
    }
}

impl<A, PV> Debug for EdgeDetails<A, PV> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<A, PV> Ord for EdgeDetails<A, PV> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.Nsa, &self.Qsa, &self.Psa, &self.Usa, &self.cpuct).partial_cmp(&(
            other.Nsa,
            &other.Qsa,
            &other.Psa,
            &other.Usa,
            &other.cpuct,
        )) {
            Some(ordering) => ordering,
            None => {
                panic!(
                    "Could not compare: {:?} to {:?}",
                    (self.Nsa, &self.Qsa, &self.Psa, &self.Usa, &self.cpuct),
                    (other.Nsa, &other.Qsa, &other.Psa, &other.Usa, &other.cpuct)
                );
            }
        }
    }
}

impl<A, PV> PartialOrd for EdgeDetails<A, PV> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, PV> Eq for EdgeDetails<A, PV> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Nsa() {
        let puct_greater = EdgeDetails {
            Nsa: 2,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            Nsa: 1,
            Qsa: 2.0,
            Psa: 2.0,
            Usa: 2.0,
            Msa: 2.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Qsa() {
        let puct_greater = EdgeDetails {
            Nsa: 1,
            Qsa: 2.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 2.0,
            Usa: 2.0,
            Msa: 2.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Psa() {
        let puct_greater = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 2.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 2.0,
            Msa: 2.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Usa() {
        let puct_greater = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 2.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 2.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_cpuct() {
        let puct_greater = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 2.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            Nsa: 1,
            Qsa: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            M: 0.0,
            game_length: 1.0,
            moves_left_score: 1.0,
            cpuct: 1.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }
}
