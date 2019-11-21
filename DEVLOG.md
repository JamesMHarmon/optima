Run-5

11-9-19: Starting fresh run w/ 64x5 and 64x5 networks. Removed cache since it was giving faulty results. Added traps to input planes. Doing run over run-4 due to the smaller network giving much better performance. Also lower memory constraints on the GPU.

Run-6

11-10?-19: Starting run-6. Disabling Pass action to reduce abuse of this action. Increasing max game length from 128 to 512. 

11-13-19: Lowering number_of_games_per_net from 16k to 12k since games per net is around 40k. Lowering position samples from 0.0512 to 0.0256. Raising learning rate to 0.1

11-16-19: model 30: Lowering moving_window_size from 250k to 125k. Lowering position samples from 0.0256 to 0.0128

11-17-19: model 40: Introduced new always training mode. increased moving_window_size from 125k to 250k. Increased position samples from 0.0128 to 0.0256.

11-18-19: model 0: Starting run-7. Bug fixed w/ push as well as an issue w/ the model only being update w/ the last batch of data.

11-20-19: model 20: Updating remote to kill games immediately on a new model as well as discard games that are too long.


