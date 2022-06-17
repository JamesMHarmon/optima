from replay_buffer import ReplayBuffer
import pprint

replay_buffer = ReplayBuffer(games_dir="../Arimaa_runs/games", min_visits=400, mode="play")
sample = replay_buffer.sample(2048, start_idx=0, end_idx=2047)

pprint.pprint(sample)
pprint.pprint(replay_buffer.games())
