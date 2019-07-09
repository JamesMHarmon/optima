# Alpha Zero General

## Features
* **General** - Alpha Zero General exposes trait for any game to be able to plug into.
* **Performant** - Alpha Zer
* **Distributed**
* **Multi-Player** - Most implementations of AZ only allow for two player games.

## TODO:
* Allow for multiple players
* Reimpliment CPUCT and Temp functions
* Add self play evaluation/tournament
* Make learning distributed
* Break into a workspace
* HELP!: Convert from Keras to rust-tensorflow to remove python dependency
* Add WASM support
* Add better debug printing
* Add better error handling
* Add max moves
* Document Code
* Document READMEs
* Add instructions on how to get it working
* Dockerize

## References

* https://github.com/tensorflow/rust
* https://www.groundai.com/project/hyper-parameter-sweep-on-alphazero-general/1



* Create server client
            
            
            // https://github.com/evg-tyurin/alpha-nagibator/blob/master/MCTS.py
            // p = (1-e) * p + e * noise[i] // + 1
            // Checkers
            // 'dirAlpha': 0.3,
            // 'epsilon': 0.25, 


            // https://github.com/dylandjian/SuperGo/blob/master/models/mcts.py
            // (1 - EPS) * probas + EPS * np.random.dirichlet(np.full(dim, ALPHA))
            // ## Epsilon for Dirichlet noise
            // EPS = 0.25
            // ## Alpha for Dirichlet noise
            // ALPHA = 0.03

            // https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py
            // nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
            // (1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
            // EPSILON = 0.2
            // ALPHA = 0.8


            // https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
            // p = (1-x) * d + x * p
            // In the paper, x is set to 0.75
            // It’s hard to know how to optimally extrapolate, but a reasonable first guess looks to be choosing ɑ = 10/n
            // Since there are approximately four legal moves in a Connect Four position, this gives us ɑ=2.5. It’s true that this is greater than one while the rest are less, and yet this number seemed to do well in our testing. With a little playing around, we found that 1.75 did even better.
            // UPDATE: while early versions of our training did best with a=1.75, we ultimately settled on a=1.0 as the optimal value for our training.
