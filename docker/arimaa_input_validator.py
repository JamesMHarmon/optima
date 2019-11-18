import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt

POSITION = 0
FILE_NAME = '/tmp/sample.csv'

input_h = 8
input_w = 8
input_c = 50
input_size = input_h * input_w * input_c
channels_per_step = 12
steps = 4
pieces = np.asarray([" ", "E","M","H","D","C","R","e","m","h","d","c","r"])

output_h = 8
output_w = 7
outputs = 4
output_names = ["n","e","s","w"]

sample_data = genfromtxt(FILE_NAME, delimiter=',')
X = sample_data[:,:input_size].reshape(-1, input_h, input_w, input_c)
yp = sample_data[:,input_size:-1]
yv = sample_data[:,-1]

X = X[POSITION]
yp = yp[POSITION]
yv = yv[POSITION]

def getChannel(arr, channel_idx):
    return arr[:,:,channel_idx]

for step in range(0, steps):
    channels = []
    for i in range(0, channels_per_step):
        print("Channel: " + str(i))
        channel = getChannel(X, (step * channels_per_step) + i)
        print(channel)
        channel = np.where(channel==1, i + 1, 0)
        channels.append(channel)
        print("")

    flattened = np.zeros((input_h, input_w))
    for i in range(0, channels_per_step):
        flattened += channels[i]

    print("Flattened Step")
    flattened = flattened.astype("int")
    print(pieces[flattened])

print("Step Plane")
print("")
step_plane = getChannel(X, (steps * channels_per_step))
print(step_plane)

print("")
moves = yp[:-1]
for i in range(0,outputs):
    print("Policies: " + output_names[i])
    (output_w,output_h) = (7,8) if i % 2 == 0 else (8,7)
    print(moves.reshape(outputs,output_w,output_h)[i])
    print("")
    
print("")
print("Pass: ", yp[-1])

print("")
print("Value: ", yv)