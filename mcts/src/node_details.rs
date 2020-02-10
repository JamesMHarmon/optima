use std::cmp::Ordering;
use std::fmt::{self,Display,Formatter};

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct NodeDetails<A> {
    pub visits: usize,
    pub children: Vec<(A, PUCT)>
}

impl<A: Display> Display for NodeDetails<A> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let actions = format!("[{}]", self.children.iter().fold(String::new(), |acc, (a, puct)| acc + &format!("\n\t(A: {}, {}),", a, puct)));

        write!(f, "V: {visits}, Actions: {actions}",
            visits = self.visits,
            actions = actions
        )
    }
}

#[derive(Debug,PartialEq)]
#[allow(non_snake_case)]
pub struct PUCT {
    pub Nsa: usize,
    pub Qsa: f32,
    pub logitQ: f32,
    pub Psa: f32,
    pub Usa: f32,
    pub Msa: f32,
    pub moves_left: f32,
    pub cpuct: f32,
    pub PUCT: f32,
}

impl Display for PUCT {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Nsa: {Nsa}, Qsa: {Qsa}, Qlogit: {logitQ}, Msa: {Msa}, Psa: {Psa}, Usa: {Usa}, cpuct: {cpuct}, moves_left: {moves_left}, PUCT: {PUCT}",
            Nsa = self.Nsa,
            Qsa = self.Qsa,
            logitQ = self.logitQ,
            Psa = self.Psa,
            Usa = self.Usa,
            Msa = self.Msa,
            moves_left = self.moves_left,
            cpuct = self.cpuct,
            PUCT = self.PUCT,
        )
    }
}

impl Ord for PUCT {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.Nsa, &self.Qsa, &self.Psa, &self.Usa, &self.cpuct).partial_cmp(&(other.Nsa, &other.Qsa, &other.Psa, &other.Usa, &other.cpuct)) {
            Some(ordering) => ordering,
            None => {
                panic!("Could not compare: {:?} to {:?}", (self.Nsa, &self.Qsa, &self.Psa, &self.Usa, &self.cpuct), (other.Nsa, &other.Qsa, &other.Psa, &other.Usa, &other.cpuct));
            }
        }
    }
}

impl PartialOrd for PUCT {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PUCT { }

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Nsa() {
        let puct_greater = PUCT {
            Nsa: 2,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 1.0,
            PUCT: 1.0,
        };

        let puct_less = PUCT {
            Nsa: 1,
            Qsa: 2.0,
            logitQ: 1.0,
            Psa: 2.0,
            Usa: 2.0,
            Msa: 2.0,
            moves_left: 2.0,
            cpuct: 2.0,
            PUCT: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Qsa() {
        let puct_greater = PUCT {
            Nsa: 1,
            Qsa: 2.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 1.0,
            PUCT: 1.0,
        };

        let puct_less = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 2.0,
            Psa: 2.0,
            Usa: 2.0,
            Msa: 2.0,
            moves_left: 2.0,
            cpuct: 2.0,
            PUCT: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Psa() {
        let puct_greater = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 2.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 1.0,
            PUCT: 1.0,
        };

        let puct_less = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 2.0,
            Msa: 2.0,
            moves_left: 2.0,
            cpuct: 2.0,
            PUCT: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_Usa() {
        let puct_greater = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 2.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 1.0,
            PUCT: 1.0,
        };

        let puct_less = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 2.0,
            PUCT: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_node_details_ordering_cpuct() {
        let puct_greater = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 2.0,
            PUCT: 1.0,
        };

        let puct_less = PUCT {
            Nsa: 1,
            Qsa: 1.0,
            logitQ: 1.0,
            Psa: 1.0,
            Usa: 1.0,
            Msa: 1.0,
            moves_left: 1.0,
            cpuct: 1.0,
            PUCT: 2.0,
        };

        assert_eq!(puct_less.cmp(&puct_greater), Ordering::Less);
        assert_eq!(puct_greater.cmp(&puct_less), Ordering::Greater);
    }
}

