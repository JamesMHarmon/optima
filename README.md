# Alpha Zero General

## Features
* **General** - Alpha Zero General exposes trait for any game to be able to plug into.
* **Performant** - Alpha Zer
* **Distributed**
* **Multi-Player** - Most implementations of AZ only allow for two player games.

## TODO:
* Allow for multiple players
* Make learning distributed
* Add WASM support
* Add better debug printing
* Add better error handling
* Add max moves
* Document Code
* Document READMEs
* Add instructions on how to get it working
* Convert reqwest http calls to use std::futures
* Mask policy head
* Create multi-threaded MCTS


## References

* https://github.com/tensorflow/rust
* https://www.groundai.com/project/hyper-parameter-sweep-on-alphazero-general/1
* https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a
* https://gist.github.com/erenon/cb42f6656e5e04e854e6f44a7ac54023
* http://blog.lczero.org/2018/12/alphazero-paper-and-lc0-v0191.html

pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init


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

learning rates
// - 0.1
// - 0.01
// - 0.001
// - 0.0001


until ./quoridor run -g "Connect4" -r "Run-1"; do echo "Server 'myserver' crashed with exit code $?.  Respawning.." >&2;     sleep 1; done

until docker run --rm     --runtime=nvidia     -p 8501:8501     --mount type=bind,source="$(pwd)/Quoridor_runs/run-1",target=/models     --mount type=bind,source="$(pwd)/Quoridor_runs/run-1/models.config",target=/models/models.config     --env-file "$(pwd)/Quoridor_runs/run-1/env.list"     -t     tensorflow/serving:latest-gpu --model_config_file=/models/models.config; do echo "Server 'myserver' crashed with exit code $?.  Respawning.." >&2;     sleep 10; done


## Commands

sudo docker run --rm \
    --runtime=nvidia \
    -p 8501:8501 \
    --mount type=bind,source=$(pwd)/Connect4_runs/run-1,target=/models \
    --mount type=bind,source=$(pwd)/Connect4_runs/run-1/models.config,target=/models/models.config \
    --env-file $(pwd)/Connect4_runs/run-1/env.list \
    -t \
    tensorflow/serving:latest-gpu --model_config_file=/models/models.config

https://github.com/tensorflow/serving/issues/1077

sudo docker run --rm \
    --runtime=nvidia -it \
    -v $(pwd):/tmp tensorflow/tensorflow:latest-gpu \
    /usr/local/bin/saved_model_cli convert \
    --dir /tmp/export_model/1 \
    --output_dir /tmp/export_model_trt/1 \
    --tag_set serve \
    tensorrt --precision_mode FP16 --max_batch_size 512

Time Elapsed: 0.20h, Number of Games Played: 1024

curl -d '{"instances": [[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]]}'     -X POST http://localhost:8501/v1/models/exported_models/versions/2:predict



@Alvaro.  Here is a repost: "
So do I understand this correctly? 4096 positions are sampled randomly from the window of 1M games and any number of positions. These 4096 positions are feed to the network with the outcome. Then again 4096 randomly selected positions are selected and feed to the network and training it. This is repeated 250 times. After that the new network is uploaded.
During selection of 4096 games, 250 times, the set of the games and positions in the window is "frozen", nothing is changed?
So each time a new network is generated, it has been trained with 250 * 4096 = 1 024 000 positions randomly selected from the window of 1 000 000 games (and unknown number of positions), while no altering of the window occurred.


---------------------------
aartToday at 2:39 PM
as far as I know that is all true



fischerandomToday at 2:55 PM
And then after the new network has been uploaded, all positions from all self-play games generated by all clients sent to the server is pushed into the game pool window of 1 000 000 games, and at the same time the oldest games in the 1M window are removed. So if 32000 games are pushed into the window, then first the 32000 oldest games are removed from the window?
aartToday at 2:56 PM
I think so


export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64


GPU vs CPU
Change train.Dockerfile to use cpu
change flag when running docker to not use environment = nvidia.

Update Display to show shorthand moves "f4"

Why is there a bias towards p2 in the Quoridor_run-1_00003 results