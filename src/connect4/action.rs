use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Action {
    DropPiece(u64)
}

#[derive(Debug)]
pub struct ValidActions(u64);
