pub trait ChannelExt<T> {
    // Blocks the channel for at least one item and then attempts to receive additional items up to the limit specified.
    fn recv_up_to(&self, limit: usize) -> Vec<T>;
}

impl<T> ChannelExt<T> for crossbeam::channel::Receiver<T> {
    fn recv_up_to(&self, limit: usize) -> Vec<T> {
        let mut states_to_analyse = Vec::with_capacity(limit);

        if let Ok(state_to_analyze) = self.recv() {
            states_to_analyse.push(state_to_analyze);
        }

        while let Ok(state_to_analyze) = self.try_recv() {
            states_to_analyse.push(state_to_analyze);

            if states_to_analyse.len() == limit {
                break;
            }
        }

        states_to_analyse
    }
}
