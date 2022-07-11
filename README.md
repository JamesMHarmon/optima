# Alpha Zero General

AZG is a program designed to master a variety of abstract strategy board games from scratch. AZG achieves this mastery by a process known as reinforcement learning. AZG starts off by playing random moves to explore its environment, then it trains a neural network on that newly acquired information. The newly trained neural network is then used to play additional games. This process is repeated until the AZG has mastered the game. AZG is based on DeepMind's Alpha Zero and implements some novel improvements pioneered by lczero, lc0 and KataGo.

## Features

* **General** - AZG exposes traits which allows AZG to play a variety of games.
* **Performant** - By utilizing rust's async await, thousands of games can be played in parrallel. This allows for games to be learned w/ much less computation required than the original Alpha Zero.
* **Multi-Player** - Most implementations of Alpha Zero only allow for two player games. AZG abstracts out the players and allows for games of N players to be played.
* **Non-alternating actions** - Most implementations of Alpha Zero only allow for games where turns are alternating. AZG abstracts out the actions and allows for games which require players to take multiple actions in a row.
* **Parallelism** - AZG implements a concept known as virtual loss which allows many threads to search the game tree simultaniously. This allows for high utilization of the machines available resources.
* **Tensorflow & TensorRT** - Tensorflow combined with TensorRT allows for relatively fast neural network inference. Any NVIDIA RTX series card will allow for the best experience in self-learning and play speeds.
* **Componentized** - One of the goals of AZG is to not only allow an assortment of games to be implemented, but also to allow for the different pieces in the system to be changed. For example, a model can be changed to use pytorch vs tensorflow. This is to allow for greater flexibility and experimentation of novel ideas. Furthermore, adhering to rust's idea of zero cost abstractions, this approach does not impact performance in any way.

## Setup and Play

AZG includes some pre-trained networks that allow for immediate play. To learn from scratch, reference the section [Self-Learn].

It is recommended to try running the CPU version initially due to the relative ease of setup. The GPU version requires some additional dependencies.

### CPU

```bash
# clone the repo
git clone AZG

# Build the package
cd ./AZG && cargo build --release

# Run the client
./target/release/self-learn-client play game -g <C4|Quoridor|Arimaa> -r run-1
```

### GPU

* Change train.Dockerfile to use cpu
* change flag when running docker to not use environment = nvidia.

## Games

AZG can play a variety of games as well as allows support for implementing additional games beyond what is listed below.

### Arimaa

### Quoridor

### Connect4

## Self-Learn

AZG

## Install

[Installation Instructions](./INSTALL.md)

## TODO

* Update MCTS to search invalid repetition moves during self learn to prevent bias in moves that should be valid. Filter these moves out during the choose action step.

## FAQ

Q: I get the error `Op type not registered 'FusedBatchNormV3' in binary`
A: Use `cargo build --release` as opposed to `cargo run`. Otherwise build `libtensorflow.so` for your specific machine.

## References

* https://github.com/tensorflow/rust
* https://www.groundai.com/project/hyper-parameter-sweep-on-alphazero-general/1
* https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a
* https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023
* http://blog.lczero.org/2018/12/alphazero-paper-and-lc0-v0191.html
