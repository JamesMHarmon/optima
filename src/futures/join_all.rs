use std::fmt::Debug;
use std::task::Poll;
use std::task::Context;
use std::marker::Unpin;
use std::pin::Pin;
use std::future::Future;

pub struct JoinAllFuture<T,O>
    where T: Future<Output=O> + Unpin
{
    pending: Vec<(usize, T)>,
    complete: Vec<Option<O>>
}

impl<T,O> Future for JoinAllFuture<T,O>
    where
        T: Future<Output=O> + Unpin,
        O: Unpin + Debug
{
    type Output = Vec<O>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let s = self.get_mut();
        if s.pending.len() >= 1 {
            let mut results: Vec<(usize,usize,O)> = Vec::new();
            let pending = &mut s.pending;
            for (pending_index, (complete_index, f)) in pending.iter_mut().enumerate() {
                if let Poll::Ready(r) = Future::poll(Pin::new(f), cx) {
                    results.push((pending_index, *complete_index,r));
                }
            }

            results.reverse();
            for (pending_index, complete_index,o) in results {
                pending.remove(pending_index);
                s.complete.get_mut(complete_index).unwrap().replace(o);
            }
        }

        if s.pending.len() == 0 {
            let outputs: Vec<O> = s.complete.iter_mut().map(|v| v.take().unwrap()).collect();
            return Poll::Ready(outputs);
        }

        Poll::Pending
    }
}


impl<T,O> JoinAllFuture<T,O>
    where T: Future<Output=O> + Unpin
{
    fn new(futures: Vec<T>) -> Self {
        Self {
            complete: (0..futures.len()).map(|_| None).collect(),
            pending: futures.into_iter().enumerate().collect(),
        }
    }
}

pub fn join_all<T,O>(futures: Vec<T>) -> JoinAllFuture<T,O>
    where T: Future<Output=O> + Unpin
{
    JoinAllFuture::<T,O>::new(futures)
}