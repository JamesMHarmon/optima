use std::sync::Mutex;

pub struct IncrementingMap<T> {
    buckets: Vec<Bucket<T>>,
    capacity: usize
}

struct Bucket<T> {
    items: Mutex<Vec<(usize, T)>>
}

impl<T> Bucket<T> {
    fn new() -> Self {
        Self { 
            items: Mutex::new(Vec::new())
        }
    }
}

impl<T> IncrementingMap<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buckets = Vec::with_capacity(capacity);

        for _ in 0..capacity {
            buckets.push(Bucket::new());
        }

        Self {
            capacity,
            buckets
        }
    }

    pub fn insert(&self, id: usize, item: T) {
        let index = self.get_index(id);
        let bucket_lock = self.buckets[index].items.lock();
        let bucket = &mut bucket_lock.unwrap();
        bucket.push((id, item));
    }

    pub fn remove(&self, id: usize) -> Option<T> {
        let index = self.get_index(id);
        let bucket_lock = self.buckets[index].items.lock();
        let bucket = &mut bucket_lock.unwrap();
        match bucket.iter().position(|(pid, _)| *pid == id) {
            Some(bucket_index) => {
                let (_id, entry) = bucket.swap_remove(bucket_index);
                Some(entry)
            },
            None => None
        }
    }

    #[inline(always)]
    fn get_index(&self, id: usize) -> usize {
        id % self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incrementing_map_insert() {
        let map = IncrementingMap::with_capacity(2);
        let id = 2;
        let item = "Duck";

        map.insert(id, item);

        assert_eq!(map.remove(id).unwrap(), item);
    }

    #[test]
    fn test_incrementing_map_insert_multiple() {
        let map = IncrementingMap::with_capacity(2);
        let duck_id = 4;
        let duck = "Duck";

        let goose_id = 8;
        let goose = "Goose";

        map.insert(duck_id, duck);
        map.insert(goose_id, goose);

        assert_eq!(map.remove(duck_id).unwrap(), duck);
        assert_eq!(map.remove(goose_id).unwrap(), goose);
    }
}