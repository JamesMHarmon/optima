# Optima

Optima is a program that trains an agent from scratch to master a given environment as efficiently as possible. Optima achieves this mastery by a process known as model-based reinforcement learning, where a model of the environment is provided and a reward is given when the agent performs well. Optima starts off by taking random actions to explore its environment. Through this interaction with the world, it learns which of these actions led to positive outcomes and which actions led to negative outcomes. Optima then trains a deep neural network on that newly acquired information. The recently updated neural network is then used to explore its environment again, but this time with the newly required information where the agent will perform slightly better than the last. This process is repeated until the Optima has mastered the environment. To act as a test bed and demonstrate its ability to master its environment, this repository includes some abstract strategy board games like Arimaa, Tak, and Quoridor. Optima is based on DeepMind's Alpha Zero and implements some novel improvements developed by similar projects Leela Zero, Lc0 and KataGo.

## Features

* **General** - Optima exposes traits which allows Optima to play a variety of games and environments. This is in contrast to other projects which aim to master a single game.
* **Performant** - By utilizing modern channel based architecture in conjunction with rust's async await, thousands of games can be played in parallel. This allows for games to be learned w/ much less computation required than the original Alpha Zero.
* **Scalable** - Optima is split into a number of processes which can each be independently scaled horizontally across multiple machines. These processes include agents interacting in the environment, training of the agents given their experience.
* **Multi-Agent** - Optima supports the training of many different agents that can interact together within an environment. This is invaluable for scenarios where multiple actors must work together towards a common goal. Note that while Optima supports the ability to have multiple agents, multi-agents will likely perform poorly depending on your environment. This is due to issues like reward attribution and the increase in variance from having additional agents. For more information, check out some papers covering multi-agent learning. https://arxiv.org/pdf/1706.02275.pdf
* **Multi-Player** - Most implementations of Alpha Zero only allow for two player games. Optima abstracts out the players and allows for Optima to learn games of anywhere from 1 to N players.
* **Non-Zero-Sum** - Optima works in zero-sum as well as in non-zero-sum environments by allowing each agent to have its own set of rewards. This means that agents can choose to work both cooperatively and/or competitively as the agent attempts to maximize it's own reward.
* **Non-alternating actions** - Most implementations of Alpha Zero only allow for games where turns are alternating. Optima abstracts out the actions and allows for games which require players to take multiple actions in a row and for scenarios where turn order is dynamic.
* **Parallelism** - Optima implements a concept known as virtual loss which allows many threads to search the game tree simultaneously. This allows for high utilization of the machines available resources.
* **Tensorflow & TensorRT** - Tensorflow combined with TensorRT allows for relatively fast neural network inference. Any NVIDIA RTX series card will allow for the best experience in self-learning and play speeds.
* **Componentized** - One of the goals of Optima is to not only allow an assortment of games to be implemented, but also to allow for the different pieces in the system to be changed. For example, a model can be changed to use pytorch vs tensorflow. This is to allow for greater flexibility and experimentation of novel ideas while having little to no impact on performance.

## Improvements

* **Arena** - An arena is introduced which allows many candidate agents to compete against other agents for a chance to be used for self-play. This allows for many different network sizes and architectures to be implemented in a single run. In addition, agents are matched against previous versions of themselves. This prevents the agent from getting stuck in a local optimum. By continually comparing the performance of the agent against previous versions, this ensures that only the most fit agent is selected for further training. This helps the agent to overcome potential issues like plateauing, regressing, or catastrophic collapse as an agent won't be selected until it overcomes its temporary shortfall. Due to the off-policy nature of the algorithms, while these candidate agents are not actively selected for self-play, they are still learning through shadowing of their champion counterparts.

* **Ancillary Heads** - While Alpha Zero incorporated a single network with two output heads, policy and value, Optima has introduced additional outputs as well as improvements. First, the policy head is shortened and has its dense layers removed, this allows the policy to work directly with the logits from the convolutional layers. This means that the policy head is more general and spatially aware as all outputs, regardless of location, utilize the same weights. Second, additional heads like 

## Performance

Optima is designed with performance as a first class concern. Deep Learning and especially Reinforcement Learning are known to be extremely computationally expensive. Some have estimated a single training run of Alpha Zero could require 500k GPU Hours costing appx. $2.9 Million USD in compute. Other similar open source projects may require around 20k GPU hours per run costing appx. $20k-$60k USD in compute. Optima has mastered these games in less than 5k GPU hours. Achieving perfect play on a test-bed like Connect4 is done on a single GPU in under an hour.

This performance is achieved through a variety of mechanisms. The first being that Optima supports multi-agent, this allows small networks that can act as weak learners which learn faster and are much less costly to infer than their strong learner counter parts. Large nets acting as strong learners will not be selected until their strength has justified their inference cost. Second, improvements introduced by KataGo for reducing the number of inferences per game have been implemented. By reducing the cost per game, the agents can play . There may be some further reduction through Gumbel Zero methodologies.

## Setup and Play

Optima includes some pre-trained networks that allow for immediate play. To learn from scratch, reference the section [Self-Learn].

It is recommended to try running the CPU version initially due to the relative ease of setup. The GPU version requires some additional dependencies.

### CPU

```bash
# clone the repo
git clone Optima

# Build the package
cd ./Optima && cargo build --release

# Run the client
./target/release/self-learn-client play game -g <C4|Quoridor|Arimaa> -r run-1
```

### GPU

* Change train.Dockerfile to use cpu
* change flag when running docker to not use environment = nvidia.

## Games

Optima can play a variety of games as well as allows support for implementing additional games beyond what is listed below.

### Arimaa

http://arimaa.com/arimaa/mwiki/index.php/List_of_bots

### Quoridor

### Connect4

## Self-Learn

Optima

## Use

Please note that it is **NOT** recommended to use Optima. Optima is created for personal use and is not well supported. This repository is made open source for visibility and collaboration as well as for my own learnings. There are no plans or guarantees to maintain this in the future. There are much better libs available which are more feature rich, more robust and well tested, better maintained and have communities of support around them. Please use one of those.

## Install

[Installation Instructions](./INSTALL.md)

## FAQ

Q: I get the error `Op type not registered 'FusedBatchNormV3' in binary`
A: Optima is referencing the incorrect version of TF on your machine. Either yse `cargo build --release` as opposed to `cargo run`. Otherwise, build `libtensorflow.so` for your specific machine and provide the correct path with environment variables.

## References

* https://github.com/tensorflow/rust
* https://www.groundai.com/project/hyper-parameter-sweep-on-alphazero-general/1
* https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a
* https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023
* http://blog.lczero.org/2018/12/alphazero-paper-and-lc0-v0191.html
