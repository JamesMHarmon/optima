#[derive(Debug, Clone)]
pub enum Action {
    DropPiece(u64)
}

#[derive(Debug)]
pub struct ValidActions(u64);
