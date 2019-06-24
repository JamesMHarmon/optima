use rand::prelude::{SeedableRng, StdRng};
use rand::RngCore;
use uuid::Uuid;

pub fn create_rng_from_uuid(uuid: Uuid) -> impl RngCore {
    let uuid_bytes: &[u8; 16] = uuid.as_bytes();
    let mut seed = [0; 32];
    seed[..16].clone_from_slice(uuid_bytes);
    seed[16..32].clone_from_slice(uuid_bytes);

    let seedable_rng: StdRng = SeedableRng::from_seed(seed);

    seedable_rng
}
