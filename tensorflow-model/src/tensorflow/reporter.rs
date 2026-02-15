use common::TranspositionTable;
use log::info;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio::time;

pub struct Reporter<Te> {
    alive: Arc<AtomicBool>,
    inner: Arc<ReporterInner<Te>>,
}

impl<Te> Reporter<Te>
where
    Te: Send + 'static,
{
    pub fn new(transposition_table: Arc<Option<TranspositionTable<Te>>>) -> Self {
        let alive = Arc::new(AtomicBool::new(true));
        let inner = Arc::new(ReporterInner::new(transposition_table));

        Self::spawn_timer(alive.clone(), inner.clone());

        Self { alive, inner }
    }

    pub fn set_batch_size(&self, analysis_len: usize) {
        self.inner
            .min_batch_size
            .fetch_min(analysis_len, Ordering::Relaxed);

        self.inner
            .max_batch_size
            .fetch_max(analysis_len, Ordering::Relaxed);
    }

    pub fn set_terminal(&self) {
        self.inner
            .num_nodes_analysed
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_cache_hit(&self) {
        self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_cache_miss(&self) {
        self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// A predict request was satisfied immediately from TT/cache.
    pub fn set_predict_cache_or_tt(&self) {
        self.inner
            .predict_cache_or_tt
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A predict request was coalesced onto an existing in-flight inference.
    pub fn set_predict_in_flight(&self) {
        self.inner.predict_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// A predict request required a new inference to be queued.
    pub fn set_predict_needs_infer(&self) {
        self.inner
            .predict_needs_infer
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_analyzed_node(&self) {
        self.inner
            .num_nodes_analysed
            .fetch_add(1, Ordering::Relaxed);
    }

    fn spawn_timer(alive: Arc<AtomicBool>, inner: Arc<ReporterInner<Te>>) {
        tokio::task::spawn(async move {
            let duration = Duration::from_secs(5);
            let mut interval = time::interval(duration);
            loop {
                interval.tick().await;

                if !alive.load(Ordering::Relaxed) {
                    break;
                }

                inner.report(duration);
            }
        });
    }
}

impl<Te> Drop for Reporter<Te> {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

pub struct ReporterInner<Te> {
    last_report_had_nodes: AtomicBool,
    num_nodes_analysed: AtomicUsize,
    min_batch_size: AtomicUsize,
    max_batch_size: AtomicUsize,
    cache_misses: AtomicUsize,
    cache_hits: AtomicUsize,
    predict_cache_or_tt: AtomicUsize,
    predict_in_flight: AtomicUsize,
    predict_needs_infer: AtomicUsize,
    transposition_table: Arc<Option<TranspositionTable<Te>>>,
}

impl<Te> ReporterInner<Te> {
    fn new(transposition_table: Arc<Option<TranspositionTable<Te>>>) -> Self {
        let last_report_had_nodes = AtomicBool::new(false);
        let num_nodes_analysed = AtomicUsize::new(0);
        let min_batch_size = AtomicUsize::new(usize::MAX);
        let max_batch_size = AtomicUsize::new(0);
        let cache_misses = AtomicUsize::new(0);
        let cache_hits = AtomicUsize::new(0);
        let predict_cache_or_tt = AtomicUsize::new(0);
        let predict_in_flight = AtomicUsize::new(0);
        let predict_needs_infer = AtomicUsize::new(0);

        Self {
            last_report_had_nodes,
            num_nodes_analysed,
            min_batch_size,
            max_batch_size,
            cache_misses,
            cache_hits,
            predict_cache_or_tt,
            predict_in_flight,
            predict_needs_infer,
            transposition_table,
        }
    }

    fn report(&self, elapsed: Duration) {
        let elapsed_millis = elapsed.as_millis();
        let transposition_hits = self.take_transposition_hits();
        let num_infer_nodes = self.take_num_nodes_analysed();
        let num_transpo_nodes =
            transposition_hits.map_or(0, |(_entries, _capacity, hits, _misses)| hits);
        let (min_batch_size, max_batch_size) = self.take_min_max_batch_size();
        let (predict_cache_or_tt, predict_in_flight, predict_needs_infer) =
            self.take_predict_stats();
        let infer_nps = num_infer_nodes as f32 * 1000.0 / elapsed_millis as f32;
        let total_nps =
            (num_infer_nodes + num_transpo_nodes) as f32 * 1000.0 / elapsed_millis as f32;

        if self.last_report_had_nodes.load(Ordering::Relaxed) || total_nps > 0.0 {
            info!(
                "NPS: {total_nps:.2}, Infered NPS: {infer_nps:.2}, Min Batch Size: {min_batch_size}, Max Batch Size: {max_batch_size}",
                total_nps = total_nps,
                infer_nps = infer_nps,
                min_batch_size = min_batch_size,
                max_batch_size = max_batch_size
            );
            if let Some((entries, capacity, hits, misses)) = transposition_hits {
                info!(
                    "Hits: %{:.2}, Cache Hydration: %{:.2}, Entries: {}, Capacity: {}",
                    if hits > 0 {
                        (hits as f32 / (hits + misses) as f32) * 100f32
                    } else {
                        0f32
                    },
                    (entries as f32 / capacity as f32) * 100f32,
                    entries,
                    capacity
                );
            }

            let total_predict = predict_cache_or_tt + predict_in_flight + predict_needs_infer;
            if total_predict > 0 {
                let saved = predict_cache_or_tt + predict_in_flight;
                let saved_pct = saved as f32 / total_predict as f32 * 100.0;
                let cache_pct = predict_cache_or_tt as f32 / total_predict as f32 * 100.0;
                let inflight_pct = predict_in_flight as f32 / total_predict as f32 * 100.0;
                let infer_pct = predict_needs_infer as f32 / total_predict as f32 * 100.0;

                info!(
                    "Predict savings: %{saved_pct:.2} (TT/cache: %{cache_pct:.2}, in-flight: %{inflight_pct:.2}, inferred: %{infer_pct:.2})",
                    saved_pct = saved_pct,
                    cache_pct = cache_pct,
                    inflight_pct = inflight_pct,
                    infer_pct = infer_pct,
                );
            }

            self.last_report_had_nodes
                .store(total_nps > 0.0, Ordering::Relaxed);
        }
    }

    fn take_num_nodes_analysed(&self) -> usize {
        self.num_nodes_analysed.swap(0, Ordering::Relaxed)
    }

    fn take_min_max_batch_size(&self) -> (usize, usize) {
        (
            self.min_batch_size.swap(usize::MAX, Ordering::Relaxed),
            self.max_batch_size.swap(0, Ordering::Relaxed),
        )
    }

    fn take_transposition_hits(&self) -> Option<(usize, usize, usize, usize)> {
        self.transposition_table
            .as_ref()
            .as_ref()
            .map(|transposition_table| {
                (
                    transposition_table.num_entries(),
                    transposition_table.capacity(),
                    self.cache_hits.swap(0, Ordering::Relaxed),
                    self.cache_misses.swap(0, Ordering::Relaxed),
                )
            })
    }

    fn take_predict_stats(&self) -> (usize, usize, usize) {
        (
            self.predict_cache_or_tt.swap(0, Ordering::Relaxed),
            self.predict_in_flight.swap(0, Ordering::Relaxed),
            self.predict_needs_infer.swap(0, Ordering::Relaxed),
        )
    }
}
