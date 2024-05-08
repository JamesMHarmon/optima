pub struct DirichletOptions {
    pub epsilon: f32,
}

pub struct MCTSOptions {
    pub(crate) dirichlet: Option<DirichletOptions>,
    pub(crate) fpu: f32,
    pub(crate) fpu_root: f32,
    pub(crate) temperature_visit_offset: f32,
    pub(crate) moves_left_threshold: f32,
    pub(crate) moves_left_scale: f32,
    pub(crate) moves_left_factor: f32,
    pub(crate) parallelism: usize,
}

#[allow(clippy::too_many_arguments)]
impl MCTSOptions {
    pub fn new(
        dirichlet: Option<DirichletOptions>,
        fpu: f32,
        fpu_root: f32,
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
            temperature_visit_offset,
            moves_left_threshold,
            moves_left_scale,
            moves_left_factor,
            parallelism,
        }
    }
}
