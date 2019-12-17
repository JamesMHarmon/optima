Run-5

11-9-19: Starting fresh run w/ 64x5 and 64x5 networks. Removed cache since it was giving faulty results. Added traps to input planes. Doing run over run-4 due to the smaller network giving much better performance. Also lower memory constraints on the GPU.

Run-6

11-10?-19: Starting run-6. Disabling Pass action to reduce abuse of this action. Increasing max game length from 128 to 512. 

11-13-19: Lowering number_of_games_per_net from 16k to 12k since games per net is around 40k. Lowering position samples from 0.0512 to 0.0256. Raising learning rate to 0.1

11-16-19: model 30: Lowering moving_window_size from 250k to 125k. Lowering position samples from 0.0256 to 0.0128

11-17-19: model 40: Introduced new always training mode. increased moving_window_size from 125k to 250k. Increased position samples from 0.0128 to 0.0256.

Run-7

11-18-19: model 0: Starting run-7. Bug fixed w/ push as well as an issue w/ the model only being update w/ the last batch of data.

11-20-19: model 20: Updating remote to kill games immediately on a new model as well as discard games that are too long.

11-21-19: appx model 28? - Lowered cpuct from 2.5 to 1.25 - Capped game length, lowered moving window size from 250k to 125k

11-22-19: appx model 31? -

11-24-19: model 43 - Update policies to be re-normalized to sum to 1.

11-29-19: model 68 - Updated training data to be run in reverse, so that newest games are trained last.

12-5-19: model 67 - Cleared all models back to model 67. An issue was found where fpu was switched w/ fpu_root.

12-6-19: model 74 - Introducing fast visits.

12-9-19: model 163 - Added -0.9 temp offset. Pass action is now working.

12-10-19: model 190 - Started training 8x96 model

12-10-19: model 192 - Raise LR from 0.05 to 0.1

12-12-19: model 252 - Updated code to fix bug w/ noise not always being applied to root

12-12-19: model 253 - Updated to start using 8x96 net

12-13-19: model 270 - Raised moving window size from 125k to 225k

12-13-19: model 282 - Changed tensorboard to calculate loss on the most recent training data batch

12-13-19: model 318 - Lowered moving window size from 225k to 125k

12-16-19: model 344 - Reset model back to 242 after discovery an issue w/ too much dirichlet noise. Changed noise to be relative to the current number of valid actions.


