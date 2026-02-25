use tensorflow::*;

pub(super) struct TensorPool<T: TensorType> {
    tensors: Vec<Tensor<T>>,
    dimensions: [u64; 3],
}

pub(super) const BATCH_SIZE_INCR: usize = 32;

impl<T: TensorType> TensorPool<T> {
    pub(super) fn new(dimensions: [u64; 3]) -> Self {
        Self {
            tensors: vec![],
            dimensions,
        }
    }

    pub(super) fn get(&mut self, size: usize, fill: T) -> &mut Tensor<T> {
        let idx = (size - 1) / BATCH_SIZE_INCR;
        let tensors = &mut self.tensors;
        while tensors.len() <= idx {
            tensors.push(Tensor::new(&[
                ((tensors.len() + 1) * BATCH_SIZE_INCR) as u64,
                self.dimensions[0],
                self.dimensions[1],
                self.dimensions[2],
            ]));
        }

        let tensor = &mut tensors[idx];

        tensor[..].fill(fill);

        tensor
    }
}
