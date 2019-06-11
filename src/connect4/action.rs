#[derive(Debug)]
pub enum Action {
    DropPiece(u64)
}

#[derive(Debug)]
pub struct ValidActions(u64)
