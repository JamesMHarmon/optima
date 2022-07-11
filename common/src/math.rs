pub fn div_or_zero(lhs: f32, rhs: f32) -> f32 {
    if rhs == 0.0 {
        0.0
    } else {
        lhs / rhs
    }
}
