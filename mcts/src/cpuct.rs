pub trait CPUCT {
    type State;

    fn cpuct(&self, state: &Self::State, nsb: usize, is_root: bool) -> f32;
}

pub struct DynamicCPUCT<S> {
    cpuct_base: f32,
    cpuct_init: f32,
    cpuct_factor: f32,
    cpuct_root_scaling: f32,
    _marker: std::marker::PhantomData<S>,
}

impl<S> DynamicCPUCT<S> {
    pub fn new(
        cpuct_base: f32,
        cpuct_init: f32,
        cpuct_factor: f32,
        cpuct_root_scaling: f32,
    ) -> Self {
        Self {
            cpuct_base,
            cpuct_init,
            cpuct_factor,
            cpuct_root_scaling,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S> CPUCT for DynamicCPUCT<S> {
    type State = S;

    fn cpuct(&self, _: &Self::State, nsb: usize, is_root: bool) -> f32 {
        (self.cpuct_init
            + self.cpuct_factor * ((nsb as f32 + self.cpuct_base + 1.0) / self.cpuct_base).ln())
            * if is_root {
                self.cpuct_root_scaling
            } else {
                1.0
            }
    }
}

impl<S> Default for DynamicCPUCT<S> {
    fn default() -> Self {
        Self {
            cpuct_base: 19652.0,
            cpuct_init: 1.25,
            cpuct_factor: 2.4,
            cpuct_root_scaling: 1.0,
            _marker: std::marker::PhantomData,
        }
    }
}
