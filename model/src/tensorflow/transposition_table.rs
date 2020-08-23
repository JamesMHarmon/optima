use log::info;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::sync::atomic::{AtomicUsize, Ordering};

const BYTES_PER_KB: usize = 1000;
const BYTES_PER_MB: usize = BYTES_PER_KB * 1000;

pub struct TranspositionTable<Te> {
    table: Vec<TranspositionRow<Te>>,
    entries: AtomicUsize,
    capacity: usize,
    key_mask: u64,
}

impl<Te> TranspositionTable<Te> {
    pub fn new(tt_cache_size: usize) -> TranspositionTable<Te> {
        let power = calculate_tt_capacity_power::<TranspositionRow<Te>>(tt_cache_size);
        let capacity = 2u128.pow(power as u32) as usize;

        info!(
            "Initializing cache with a power of {}, a capacity of {}, and entries taking up {}MB",
            power,
            capacity,
            (std::mem::size_of::<TranspositionRow<Te>>() * capacity) / BYTES_PER_MB
        );

        let mut table = Vec::with_capacity(capacity);
        table.resize_with(capacity, Default::default);

        TranspositionTable {
            table,
            capacity,
            entries: AtomicUsize::new(0),
            key_mask: get_key_mask(power),
        }
    }

    pub fn get<'ret, 'me: 'ret>(
        &'me self,
        tranposition_key: u64,
    ) -> Option<MappedMutexGuard<'ret, Te>> {
        let idx = tranposition_key & self.key_mask;

        let guard = self.table[idx as usize].lock();

        if guard.is_none() {
            return None;
        }

        if guard.as_ref().unwrap().full_key == tranposition_key {
            Some(MutexGuard::map(guard, |e| {
                &mut e.as_mut().unwrap().tranposition
            }))
        } else {
            None
        }
    }

    pub fn set(&self, tranposition_key: u64, tranposition: Te) {
        let idx = tranposition_key & self.key_mask;

        let mut row = self.table[idx as usize].lock();

        let prev = row.replace(TranspositionEntry {
            full_key: tranposition_key,
            tranposition,
        });

        if prev.is_none() {
            self.entries.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub fn num_entries(&self) -> usize {
        self.entries.load(Ordering::SeqCst)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

fn get_key_mask(power: usize) -> u64 {
    (1 << power) - 1
}

type TranspositionRow<Te> = Mutex<Option<TranspositionEntry<Te>>>;

pub struct TranspositionEntry<Te> {
    full_key: u64,
    tranposition: Te,
}

fn calculate_tt_capacity_power<Te>(tt_cache_size_mb: usize) -> usize {
    let bytes_for_entry = std::mem::size_of::<TranspositionRow<Te>>() as u128;
    let mut max_num_entries = (tt_cache_size_mb as u128 * BYTES_PER_MB as u128) / bytes_for_entry;
    let mut capacity_power = -1;

    while max_num_entries != 0 {
        max_num_entries >>= 1;
        capacity_power += 1;
    }

    capacity_power as usize
}
