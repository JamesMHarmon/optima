use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arc_swap::ArcSwap;

pub struct RcuAppendBuffer<T> {
    inner: ArcSwap<Inner<T>>,
    writer_active: AtomicBool,
}

pub struct BufferSnapshot<T> {
    inner: Arc<Inner<T>>,
    len: usize,
}

impl<T> BufferSnapshot<T> {
    pub fn as_slice(&self) -> &[T] {
        self.inner.as_slice_len(self.len)
    }
}

impl<T> Clone for BufferSnapshot<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            len: self.len,
        }
    }
}

struct Inner<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    len: AtomicUsize,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send + Sync> Sync for Inner<T> {}

impl<T> Inner<T> {
    fn with_capacity(cap: usize) -> Self {
        let mut v = Vec::with_capacity(cap);
        v.resize_with(cap, || UnsafeCell::new(MaybeUninit::uninit()));
        Self {
            data: v.into_boxed_slice(),
            len: AtomicUsize::new(0),
        }
    }

    fn capacity(&self) -> usize {
        self.data.len()
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    fn push(&self, value: T) -> Result<(), T> {
        let idx = self.len.load(Ordering::Relaxed);
        if idx >= self.capacity() {
            return Err(value);
        }

        unsafe {
            (*self.data[idx].get()).as_mut_ptr().write(value);
        }

        self.len.store(idx + 1, Ordering::Release);
        Ok(())
    }

    fn as_slice_len(&self, len: usize) -> &[T] {
        let ptr = (self.data.as_ptr() as *const MaybeUninit<T>) as *const T;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Relaxed);
        for i in 0..len {
            unsafe {
                (*self.data[i].get()).assume_init_drop();
            }
        }
    }
}

impl<T: Clone + Send + Sync> RcuAppendBuffer<T> {
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(Inner::with_capacity(2)),
            writer_active: AtomicBool::new(false),
        }
    }

    pub fn push(&self, value: T) {
        if self
            .writer_active
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            panic!("Concurrent writers detected");
        }

        let current = self.inner.load_full();

        match current.push(value) {
            Ok(()) => {}
            Err(value) => {
                let old_len = current.len();
                let new_cap = next_capacity(current.capacity());
                let new_inner = Inner::with_capacity(new_cap);

                let ptr = (current.data.as_ptr() as *const MaybeUninit<T>) as *const T;

                for i in 0..old_len {
                    let item = unsafe { &*ptr.add(i) };
                    let _ = new_inner.push(item.clone());
                }

                let _ = new_inner.push(value);
                self.inner.store(Arc::new(new_inner));
            }
        }

        self.writer_active.store(false, Ordering::Release);
    }

    pub fn snapshot(&self) -> BufferSnapshot<T> {
        let inner = self.inner.load_full();
        let len = inner.len();
        BufferSnapshot { inner, len }
    }
}

impl<T: Clone + Send + Sync> Default for RcuAppendBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn next_capacity(current: usize) -> usize {
    match current {
        0 => 2,
        1 => 2,
        2 => 4,
        4 => 8,
        8 => 16,
        16 => 32,
        32 => 64,
        64 => 96,
        n => n + 32,
    }
}
