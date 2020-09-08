use futures_intrusive::sync::LocalManualResetEvent;
use std::cell::{Cell, UnsafeCell};
use std::future::Future;
use unreachable::UncheckedOptionExt;

#[derive(Debug)]
pub struct OnceCellAsync<T> {
    state: Cell<State>,
    inner: UnsafeCell<Option<T>>,
    waiters: LocalManualResetEvent,
}

#[derive(Clone, Copy, Debug)]
pub enum State {
    Uninitialized,
    Initializing,
    Initialized,
}

impl<T> Default for OnceCellAsync<T> {
    fn default() -> OnceCellAsync<T> {
        OnceCellAsync::new()
    }
}

impl<T> OnceCellAsync<T> {
    pub fn new() -> OnceCellAsync<T> {
        OnceCellAsync {
            inner: UnsafeCell::new(None),
            state: Cell::new(State::Uninitialized),
            waiters: LocalManualResetEvent::new(false),
        }
    }

    pub fn get(&self) -> Option<&T> {
        match self.state.get() {
            State::Uninitialized | State::Initializing => None,
            // SAFETY: Value is initialized so it is guarenteed to be some
            State::Initialized => Some(unsafe { self.get_unchecked() }),
        }
    }

    pub async fn wait(&self) -> &T {
        if let State::Uninitialized | State::Initializing = self.state.get() {
            self.waiters.wait().await;
        }

        // SAFETY: Value is initialized so it is guarenteed to be some
        unsafe { self.get_unchecked() }
    }

    pub async fn get_or_init<F>(&self, init: F) -> (&T, bool)
    where
        F: Future<Output = T>,
    {
        let mut was_initializer = false;

        if let State::Uninitialized = self.state.get() {
            self.state.set(State::Initializing);

            let result = init.await;

            let value = unsafe { &mut *self.inner.get() };
            *value = Some(result);

            self.state.set(State::Initialized);
            self.waiters.set();
            was_initializer = true;
        }

        (self.wait().await, was_initializer)
    }

    unsafe fn get_unchecked(&self) -> &T {
        (&*self.inner.get()).as_ref().unchecked_unwrap()
    }
}
