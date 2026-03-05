use std::cmp::Ordering;
use std::fmt::{self, Debug, Display, Formatter};

use common::PlayerValue;

use crate::{EdgeScore, EdgeView, WeightedMerge};

#[allow(non_snake_case)]
pub struct NodeDetails<A, SS> {
    pub visits: usize,
    pub children: Vec<EdgeDetails<A, SS>>,
}

impl<A, SS> Display for NodeDetails<A, SS>
where
    A: Display,
    SS: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let actions = format!(
            "[{}]",
            self.children.iter().fold(String::new(), |acc, details| acc
                + &format!("\n\t({}),", details))
        );

        write!(
            f,
            "V: {visits}, Actions: {actions}",
            visits = self.visits,
            actions = actions
        )
    }
}

impl<A, SS> Debug for NodeDetails<A, SS>
where
    A: Debug,
    SS: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let actions = format!(
            "[{:?}]",
            self.children.iter().fold(String::new(), |acc, details| acc
                + &format!("\n\t({:?}),", details))
        );

        write!(
            f,
            "V: {visits}, Actions: {actions}",
            visits = self.visits,
            actions = actions
        )
    }
}

#[allow(non_snake_case)]
#[derive(PartialEq)]
pub struct EdgeDetails<A, SS> {
    pub action: A,
    pub Nsa: usize,
    pub Psa: f32,
    pub Usa: f32,
    pub cpuct: f32,
    pub puct_score: f32,
    pub snapshot: SS,
    pub player_to_move: usize,
}

impl<A, SS> EdgeDetails<A, SS>
where
    SS: WeightedMerge,
{
    pub fn new(edge_view: EdgeView<A, SS>, puct_scores: EdgeScore, player_to_move: usize) -> Self {
        let snapshot = edge_view.snapshot.unwrap_or_else(SS::zero);

        Self {
            action: edge_view.action,
            Nsa: edge_view.visits as usize,
            Psa: edge_view.policy_prior,
            Usa: puct_scores.usa,
            cpuct: puct_scores.cpuct,
            puct_score: puct_scores.puct_score,
            snapshot,
            player_to_move,
        }
    }
}

#[allow(non_snake_case)]
impl<A, SS> EdgeDetails<A, SS>
where
    SS: PlayerValue,
{
    pub fn Qsa(&self) -> f32 {
        self.snapshot.player_value(self.player_to_move)
    }
}

impl<A, SS> Display for EdgeDetails<A, SS>
where
    A: Display,
    SS: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "A: {action}, Nsa: {Nsa}, puct_score: {puct_score:.3}, Psa: {Psa:.3}, Usa: {Usa:.2}, {snapshot}, cpuct: {cpuct:.2}",
            action = self.action,
            Nsa = self.Nsa,
            puct_score = self.puct_score,
            Psa = self.Psa,
            Usa = self.Usa,
            snapshot = self.snapshot,
            cpuct = self.cpuct
        )
    }
}

impl<A, SS> Debug for EdgeDetails<A, SS>
where
    A: Debug,
    SS: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "A: {action:?}, Nsa: {Nsa}, puct_score: {puct_score:.3}, Psa: {Psa:.3}, Usa: {Usa:.2}, {snapshot:?}, cpuct: {cpuct:.2}",
            action = self.action,
            Nsa = self.Nsa,
            puct_score = self.puct_score,
            Psa = self.Psa,
            Usa = self.Usa,
            snapshot = self.snapshot,
            cpuct = self.cpuct
        )
    }
}

impl<A, SS> Ord for EdgeDetails<A, SS>
where
    A: Eq,
    SS: PlayerValue + Eq,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.Nsa, self.Qsa(), &self.Psa, &self.Usa, &self.cpuct).partial_cmp(&(
            other.Nsa,
            other.Qsa(),
            &other.Psa,
            &other.Usa,
            &other.cpuct,
        )) {
            Some(ordering) => ordering,
            None => {
                panic!(
                    "Could not compare: {:?} to {:?}",
                    (self.Nsa, &self.Psa, &self.Usa, &self.cpuct),
                    (other.Nsa, &other.Psa, &other.Usa, &other.cpuct)
                );
            }
        }
    }
}

impl<A, SS> PartialOrd for EdgeDetails<A, SS>
where
    A: Eq,
    SS: PlayerValue + Eq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, SS> Eq for EdgeDetails<A, SS>
where
    A: Eq,
    SS: Eq,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[derive(PartialEq, Eq)]
    struct FakeSnapshot(i32);

    impl PlayerValue for FakeSnapshot {
        fn player_value(&self, _player: usize) -> f32 {
            self.0 as f32
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Nsa() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 2,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
            player_to_move: 1,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 2.0,
            Usa: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
            player_to_move: 1,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Qsa() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(2),
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
            player_to_move: 1,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 2.0,
            Usa: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
            player_to_move: 1,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Psa() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 2.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
            player_to_move: 1,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 2.0,
            cpuct: 2.0,
            puct_score: 2.0,
            player_to_move: 1,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Usa() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 2.0,
            cpuct: 1.0,
            puct_score: 1.0,
            player_to_move: 1,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 2.0,
            player_to_move: 1,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_cpuct() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 2.0,
            puct_score: 1.0,
            player_to_move: 1,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            snapshot: FakeSnapshot(1),
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 2.0,
            player_to_move: 1,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }
}
