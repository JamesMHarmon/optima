use crossbeam_queue::{SegQueue};

pub struct ConcurrentChunkQueue<T> {
	chunk_size: usize,
	inner: SegQueue<T>
}

impl<T> ConcurrentChunkQueue<T> {
	pub fn new(chunk_size: usize, _capacity: usize) -> Self {
		ConcurrentChunkQueue::<T> {
			chunk_size,
			inner: SegQueue::new()
		}
	}

	pub fn push(&self, entry: T) {
		self.inner.push(entry);
	}

	pub fn dequeue_chunk(&self) -> Vec<T> {
		let mut chunk: Vec<_> = Vec::with_capacity(self.chunk_size);
		let mut len = 0;
		while let Ok(state_to_analyse) = self.inner.pop() {
			chunk.push(state_to_analyse);
			len += 1;

			if len >= self.chunk_size {
				break;
			}
		}

		chunk
	}
}

