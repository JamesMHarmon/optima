use collections::RcuAppendBuffer;

#[test]
fn get_returns_stable_item_across_grow() {
    let buffer = RcuAppendBuffer::new();

    buffer.push(10);
    buffer.push(20);

    let first = buffer.get(0).expect("index 0 should exist");
    assert_eq!(*first, 10);

    // This push forces a grow (initial capacity is 2).
    buffer.push(30);
    buffer.push(40);

    // The previously obtained handle must remain valid and point to the original value.
    assert_eq!(*first, 10);

    assert_eq!(*buffer.get(1).unwrap(), 20);
    assert_eq!(*buffer.get(2).unwrap(), 30);
    assert_eq!(*buffer.get(3).unwrap(), 40);
    assert!(buffer.get(4).is_none());
}
