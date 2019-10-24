macro_rules! shift_up {
    ($exp:expr) => {
        $exp >> BOARD_WIDTH
    };
}

macro_rules! shift_down {
    ($exp:expr) => {
        $exp << BOARD_WIDTH
    };
}

macro_rules! shift_left {
    ($exp:expr) => {
        $exp >> 1
    };
}

macro_rules! shift_right {
    ($exp:expr) => {
        $exp << 1
    };
}

macro_rules! shift_pieces_up {
    ($exp:expr) => {
        shift_up!($exp & !TOP_ROW_MASK)
    };
}

macro_rules! shift_pieces_right {
    ($exp:expr) => {
        shift_right!($exp & !RIGHT_COLUMN_MASK)
    };
}

macro_rules! shift_pieces_down {
    ($exp:expr) => {
        shift_down!($exp & !BOTTOM_ROW_MASK)
    };
}

macro_rules! shift_pieces_left {
    ($exp:expr) => {
        shift_left!($exp & !LEFT_COLUMN_MASK)
    };
}
