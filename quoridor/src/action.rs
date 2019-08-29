#[derive(Clone,Debug)]
pub enum Action {
    MovePawn(u128),
    PlaceHorizontalWall(u128),
    PlaceVerticalWall(u128)
}

#[derive(Debug)]
pub struct ValidActions {
    pub vertical_wall_placement: u128,
    pub horizontal_wall_placement: u128,
    pub pawn_moves: u128
}
