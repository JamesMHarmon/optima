use std::marker::PhantomData;

use engine::GameState;

pub struct DirichletOptions {
    pub epsilon: f32,
}

pub struct MCTSOptions<S, C, T>
where
    S: GameState,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    pub(crate) dirichlet: Option<DirichletOptions>,
    pub(crate) fpu: f32,
    pub(crate) fpu_root: f32,
    pub(crate) cpuct: C,
    pub(crate) temperature: T,
    pub(crate) temperature_visit_offset: f32,
    pub(crate) moves_left_threshold: f32,
    pub(crate) moves_left_scale: f32,
    pub(crate) moves_left_factor: f32,
    pub(crate) parallelism: usize,
    _phantom_state: PhantomData<*const S>,
}

#[allow(clippy::too_many_arguments)]
impl<S, C, T> MCTSOptions<S, C, T>
where
    S: GameState,
    C: Fn(&S, usize, bool) -> f32,
    T: Fn(&S) -> f32,
{
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        fpu: f32,
        fpu_root: f32,
        cpuct: C,
        temperature: T,
        temperature_visit_offset: f32,
        moves_left_threshold: f32,
        moves_left_scale: f32,
        moves_left_factor: f32,
        parallelism: usize,
    ) -> Self {
        MCTSOptions {
            dirichlet,
            fpu,
            fpu_root,
            cpuct,
            temperature,
            temperature_visit_offset,
            moves_left_threshold,
            moves_left_scale,
            moves_left_factor,
            parallelism,
            _phantom_state: PhantomData,
        }
    }
}
