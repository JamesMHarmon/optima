use crate::node_arena::NodeId;
use crossbeam_queue::SegQueue;
use std::collections::HashSet;
use std::sync::Arc;

pub struct SearchContext {
    pub path: Vec<NodeId>,
    pub visited: HashSet<NodeId>,
}

impl SearchContext {
    pub fn new() -> Self {
        Self {
            path: Vec::with_capacity(64),
            visited: HashSet::with_capacity(64),
        }
    }

    pub fn clear(&mut self) {
        self.path.clear();
        self.visited.clear();
    }
}

impl Default for SearchContext {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SearchContextPool {
    pool: Arc<SegQueue<SearchContext>>,
}

impl SearchContextPool {
    pub fn new(capacity: usize) -> Self {
        let pool = Arc::new(SegQueue::new());

        for _ in 0..capacity {
            pool.push(SearchContext::new());
        }

        Self { pool }
    }

    pub fn acquire(&self) -> SearchContextGuard {
        let ctx = self.pool.pop().unwrap_or_default();

        SearchContextGuard {
            ctx: Some(ctx),
            pool: Arc::clone(&self.pool),
        }
    }
}

impl Clone for SearchContextPool {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
        }
    }
}

pub struct SearchContextGuard {
    ctx: Option<SearchContext>,
    pool: Arc<SegQueue<SearchContext>>,
}

impl SearchContextGuard {
    pub fn get_ref(&self) -> &SearchContext {
        self.ctx.as_ref().unwrap()
    }

    pub fn split_mut(&mut self) -> (&mut Vec<NodeId>, &mut HashSet<NodeId>) {
        let ctx = self.ctx.as_mut().unwrap();
        (&mut ctx.path, &mut ctx.visited)
    }
}

impl Drop for SearchContextGuard {
    fn drop(&mut self) {
        if let Some(mut ctx) = self.ctx.take() {
            ctx.clear();
            self.pool.push(ctx);
        }
    }
}
