pub trait Players {
    type State;

    fn players(&self, state: &Self::State) -> &[usize];
}

pub trait PlayerScore {
    type State;
    type PlayerScore;

    fn score(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerScore>;
}

pub trait PlayerResult {
    type State;
    type PlayerResult;

    fn result(&self, state: &Self::State, player_id: usize) -> Option<Self::PlayerResult>;
}
