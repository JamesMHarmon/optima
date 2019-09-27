macro_rules! shift_up {
    ($exp:expr) => {
        $exp << BOARD_WIDTH
    };
}

macro_rules! shift_down {
    ($exp:expr) => {
        $exp >> BOARD_WIDTH
    };
}

macro_rules! shift_left {
    ($exp:expr) => {
        $exp << 1
    };
}

macro_rules! shift_right {
    ($exp:expr) => {
        $exp >> 1
    };
}
