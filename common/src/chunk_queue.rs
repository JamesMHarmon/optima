use crossbeam_queue::SegQueue;

pub struct ConcurrentChunkQueue<T> {
    inner: SegQueue<T>,
}

impl<T> ConcurrentChunkQueue<T> {
    pub fn new(_chunk_size: usize, _capacity: usize) -> Self {
        ConcurrentChunkQueue::<T> {
            inner: SegQueue::new(),
        }
    }

    pub fn push(&self, entry: T) {
        self.inner.push(entry);
    }

    pub fn draining_iter(&self) -> ConcurrentChunkQueueIter<'_, T> {
        ConcurrentChunkQueueIter { inner: self }
    }
}

pub struct ConcurrentChunkQueueIter<'a, T> {
    inner: &'a ConcurrentChunkQueue<T>,
}

impl<'a, T> Iterator for ConcurrentChunkQueueIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.inner.pop().ok()
    }
}
