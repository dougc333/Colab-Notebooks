

1) Colab nvcc and nvidia-smi dont match. These versions have to match for nsys and ncu files to be generated from nvidia profiling tools. 

2) pytorch and numpy broadcasting means the (3,) convention has to be converted to (3,1) and (3,1) has to be converted to (3,) using squeeze() and unsqueeze(). 


