use std::task::Waker;
use std::task::Poll;
use std::task::Context;
use std::pin::Pin;
use std::cell::{Cell,RefCell};
use std::future::Future;

#[derive(Debug)]
pub struct WaitFor {
    wakers: RefCell<Vec<Waker>>,
    is_waiting: Cell<bool>
}

impl WaitFor {
    pub fn new() -> Self {
        Self {
            wakers: RefCell::new(vec!()),
            is_waiting: Cell::new(true)
        }
    }

    pub fn wait(&self) -> WaitForFuture {
        WaitForFuture {
            waker_index: None,
            wait_for: self
        }
    }

    pub fn wake(&self) {
        self.is_waiting.set(false);

        for w in self.wakers.borrow_mut().drain(..) {
            w.wake();
        }
    }

    fn add_waker(&self, waker: Waker) -> usize {
        let mut wakers = self.wakers.borrow_mut();
        let index = (*wakers).len();
        wakers.push(waker);

        index
    }

    fn replace_waker(&self, index: usize, waker: Waker) {
        self.wakers.borrow_mut()[index] = waker
    }
}

pub struct WaitForFuture<'a> {
    waker_index: Option<usize>,
    wait_for: &'a WaitFor
}

impl<'a> Future for WaitForFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.wait_for.is_waiting.get() {
            match self.waker_index {
                Some(index) => self.wait_for.replace_waker(index, cx.waker().clone()),
                None => {
                    let index = self.wait_for.add_waker(cx.waker().clone());
                    self.waker_index = Some(index);
                }
            }

            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}