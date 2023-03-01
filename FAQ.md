# Frequently Asked Questions

- [I get the error "Op type not registered 'FusedBatchNormV3' in binary"](#i-get-the-error-op-type-not-registered-fusedbatchnormv3-in-binary)

## I get the error "Op type not registered 'FusedBatchNormV3' in binary"

Optima is referencing the incorrect version of TF on your machine. Either yse `cargo build --release` as opposed to `cargo run`. Otherwise, build `libtensorflow.so` for your specific machine and provide the correct path with environment variables.
