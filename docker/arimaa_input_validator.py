import numpy as np
import matplotlib
import matplotlib.pyplot as plt

POSITION = 0
FILE_NAME = '/tmp/sample.npy'

input_h = 8
input_w = 8
input_c = 21
input_size = input_h * input_w * input_c
channels_per_step = 12
steps = 4
pieces = np.asarray([" ", "E","M","H","D","C","R","e","m","h","d","c","r"])

output_h = 8
output_w = 7
outputs = 4
output_size = output_h * output_w * outputs + 1 # + 1 is for pass action
output_names = ["n","e","s","w"]

moves_left_size = 128
yv_size = 1

sample_data = np.load(FILE_NAME).reshape(-1, input_size + output_size + moves_left_size + yv_size)
X = sample_data[:,0:input_size].reshape(sample_data.shape[0],input_h,input_w,input_c)
start_index = input_size
yp = sample_data[:,start_index:start_index + output_size]
start_index += output_size
yv = sample_data[:,start_index]
start_index += yv_size
ym = sample_data[:,start_index:]

X = X[POSITION]
yp = yp[POSITION]
yv = yv[POSITION]

def getChannel(arr, channel_idx):
    return arr[:,:,channel_idx]

for step in range(0, 1):
    channels = []
    for i in range(0, channels_per_step):
        print("Piece Channel: " + str(i) + " " + pieces[i])
        channel = getChannel(X, (step * channels_per_step) + i)
        print(channel)
        channel = np.where(channel==1, i + 1, 0)
        channels.append(channel)
        print("")

    flattened = np.zeros((input_h, input_w))
    for i in range(0, channels_per_step):
        flattened += channels[i]

    print("Flattened Piece Channel: " + str(step))
    flattened = flattened.astype("int")
    print(pieces[flattened])
    print("")

step_channel_idx = channels_per_step
for step in range(0, steps - 1):
    print("Step Plane " + str(step))
    step_plane = getChannel(X, (step_channel_idx + step))
    print(step_plane)
    print("")
    
valid_moves_channel_idx = step_channel_idx + 3
for plane in range(0, 5):
    valid_moves_places = ['Up', 'Right', 'Down', 'Left', 'Pass']
    print("Valid Move Plane " + valid_moves_places[plane])
    valid_move_plane = getChannel(X, (valid_moves_channel_idx + plane))
    print(valid_move_plane)
    print("")

print("Trap Plane")
trap_plane_channel_idx = valid_moves_channel_idx + 5
trap_plane = getChannel(X, trap_plane_channel_idx)
print(trap_plane)
print("")

moves = yp[:-1]
for i in range(0,outputs):
    print("Policies: " + output_names[i])
    (output_w,output_h) = (7,8) if i % 2 == 0 else (8,7)
    print(moves.reshape(outputs,output_w,output_h)[i])
    print("")
    
print("Pass: ", yp[-1])
print("")

print("Value: ", yv)
print("")

print("Policy Sum: ", yp.sum())
print("")