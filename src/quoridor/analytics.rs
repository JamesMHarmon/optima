use std::task::Poll;
use std::pin::Pin;
use std::task::Context;
use std::future::Future;

use super::super::analytics::{GameAnalytics,GameStateAnalysis};
use super::engine::{GameState};
use super::action::{Action};
use super::engine::Engine;

impl GameAnalytics for Engine {
    type Future = GameStateAnalysisFuture;
    type Action = Action;
    type State = GameState;

    fn get_state_analysis(&self, _: &GameState) -> GameStateAnalysisFuture {
        GameStateAnalysisFuture {}
    }
}

pub struct GameStateAnalysisFuture {

}

impl Future for GameStateAnalysisFuture {
    type Output = GameStateAnalysis<Action>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}
