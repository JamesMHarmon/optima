fn main() {
    cfg_aliases::cfg_aliases! {
        arimaa_game: { feature = "arimaa" },
        connect4_game: { all(feature = "connect4", not(feature = "arimaa")) },
        quoridor_game: { all(feature = "quoridor", not(feature = "arimaa"), not(feature = "connect4")) },
    }
}
