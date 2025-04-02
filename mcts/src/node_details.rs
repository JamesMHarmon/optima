use std::cmp::Ordering;
use std::fmt::{self, Debug, Display, Formatter};

use common::PropagatedValue;

#[allow(non_snake_case)]
pub struct NodeDetails<A, PV> {
    pub visits: usize,
    pub children: Vec<EdgeDetails<A, PV>>,
}

impl<A, PV> Display for NodeDetails<A, PV>
where
    A: Display,
    PV: Display,
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

impl<A, PV> Debug for NodeDetails<A, PV>
where
    A: Debug,
    PV: Debug,
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
pub struct EdgeDetails<A, PV> {
    pub action: A,
    pub Nsa: usize,
    pub Psa: f32,
    pub Usa: f32,
    pub cpuct: f32,
    pub puct_score: f32,
    pub propagated_values: PV,
}

#[allow(non_snake_case)]
impl<A, PV> EdgeDetails<A, PV>
    where PV: PropagatedValue
{
    pub fn Qsa(&self) -> f32 {
        self.propagated_values.value()
    }
}

impl<A, PV> Display for EdgeDetails<A, PV>
where
    A: Display,
    PV: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "A: {action}, Nsa: {Nsa}, puct_score: {puct_score:.3}, Psa: {Psa:.3}, Usa: {Usa:.2}, values: {propagated_values}, cpuct: {cpuct:.2}",
            action = self.action,
            Nsa = self.Nsa,
            puct_score = self.puct_score,
            Psa = self.Psa,
            Usa = self.Usa,
            propagated_values = self.propagated_values,
            cpuct = self.cpuct
        )
    }
}

impl<A, PV> Debug for EdgeDetails<A, PV>
where
    A: Debug,
    PV: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "A: {action:?}, Nsa: {Nsa}, puct_score: {puct_score:.3}, Psa: {Psa:.3}, Usa: {Usa:.2}, values: {propagated_values:?}, cpuct: {cpuct:.2}",
        action = self.action,
        Nsa = self.Nsa,
        puct_score = self.puct_score,
        Psa = self.Psa,
        Usa = self.Usa,
        propagated_values = self.propagated_values,
        cpuct = self.cpuct
    )
    }
}

impl<A, PV> Ord for EdgeDetails<A, PV>
where
    A: Eq,
    PV: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match (
            self.Nsa,
            &self.propagated_values,
            &self.Psa,
            &self.Usa,
            &self.cpuct,
        )
            .partial_cmp(&(
                other.Nsa,
                &other.propagated_values,
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

impl<A, PV> PartialOrd for EdgeDetails<A, PV>
where
    A: Eq,
    PV: Ord + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A, PV> Eq for EdgeDetails<A, PV>
where
    A: Eq,
    PV: Eq,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Nsa() {
        let puct_greater = EdgeDetails {
            action: (),
            Nsa: 2,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 2.0,
            Usa: 2.0,
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
            action: (),
            Nsa: 1,
            propagated_values: 2,
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 2.0,
            Usa: 2.0,
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
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 2.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 2.0,
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
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 2.0,
            cpuct: 1.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 1.0,
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
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 2.0,
            puct_score: 1.0,
        };

        let puct_less = EdgeDetails {
            action: (),
            Nsa: 1,
            propagated_values: 1,
            Psa: 1.0,
            Usa: 1.0,
            cpuct: 1.0,
            puct_score: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }
}
