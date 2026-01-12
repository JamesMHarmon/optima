use engine::GameEngine;

pub trait Temperature {
    type State;

    fn temp(&self, state: &Self::State) -> TempAndOffset;
}

pub struct TempAndOffset {
    pub temperature: f32,
    pub temperature_visit_offset: f32,
}

pub struct NoTemp<S> {
    pub _phantom: std::marker::PhantomData<S>,
}

impl<S> NoTemp<S> {
    pub fn new() -> Self {
        Default::default()
    }
}

impl<S> Default for NoTemp<S> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S> Temperature for NoTemp<S> {
    type State = S;

    fn temp(&self, _: &Self::State) -> TempAndOffset {
        TempAndOffset {
            temperature: 0.0,
            temperature_visit_offset: 0.0,
        }
    }
}

pub struct TemperatureConstant<S> {
    pub temperature: f32,
    pub temperature_visit_offset: f32,
    pub _phantom: std::marker::PhantomData<S>,
}

impl<S> TemperatureConstant<S> {
    pub fn new(temperature: f32, temperature_visit_offset: f32) -> Self {
        Self {
            temperature,
            temperature_visit_offset,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S> Temperature for TemperatureConstant<S> {
    type State = S;

    fn temp(&self, _: &Self::State) -> TempAndOffset {
        TempAndOffset {
            temperature: self.temperature,
            temperature_visit_offset: self.temperature_visit_offset,
        }
    }
}

pub struct TemperatureMaxMoves<'e, E> {
    pub temperature: f32,
    pub temperature_post_max_moves: f32,
    pub temperature_max_moves: usize,
    pub temperature_visit_offset: f32,
    pub engine: &'e E,
}

impl<'e, E> TemperatureMaxMoves<'e, E> {
    pub fn new(
        temperature: f32,
        temperature_post_max_moves: f32,
        temperature_max_moves: usize,
        temperature_visit_offset: f32,
        engine: &'e E,
    ) -> Self {
        Self {
            temperature,
            temperature_post_max_moves,
            temperature_max_moves,
            temperature_visit_offset,
            engine,
        }
    }
}

impl<E> Temperature for TemperatureMaxMoves<'_, E>
where
    E: GameEngine,
{
    type State = E::State;

    fn temp(&self, state: &Self::State) -> TempAndOffset {
        let move_number = self.engine.move_number(state);
        let temperature = if move_number < self.temperature_max_moves {
            self.temperature
        } else {
            self.temperature_post_max_moves
        };

        TempAndOffset {
            temperature,
            temperature_visit_offset: self.temperature_visit_offset,
        }
    }
}
