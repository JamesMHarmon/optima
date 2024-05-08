use engine::GameEngine;

pub trait Temperature {
    type State;

    fn temp(&self, state: &Self::State) -> f32;
}

pub struct TemperatureConstant<S> {
    pub temperature: f32,
    pub _phantom: std::marker::PhantomData<S>,
}

impl<S> TemperatureConstant<S> {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S> Temperature for TemperatureConstant<S> {
    type State = S;

    fn temp(&self, _: &Self::State) -> f32 {
        self.temperature
    }
}

pub struct TemperatureMaxMoves<'e, E> {
    pub temperature: f32,
    pub temperature_post_max_moves: f32,
    pub temperature_max_moves: usize,
    pub engine: &'e E,
}

impl<'e, E> TemperatureMaxMoves<'e, E> {
    pub fn new(
        temperature: f32,
        temperature_post_max_moves: f32,
        temperature_max_moves: usize,
        engine: &'e E,
    ) -> Self {
        Self {
            temperature,
            temperature_post_max_moves,
            temperature_max_moves,
            engine,
        }
    }
}

impl<'e, E> Temperature for TemperatureMaxMoves<'e, E>
where
    E: GameEngine,
{
    type State = E::State;

    fn temp(&self, state: &Self::State) -> f32 {
        let move_number = self.engine.move_number(state);
        if move_number < self.temperature_max_moves {
            self.temperature
        } else {
            self.temperature_post_max_moves
        }
    }
}
