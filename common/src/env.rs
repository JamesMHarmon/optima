pub fn get_env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{} must be a valid number", key))
        })
        .ok()
}
