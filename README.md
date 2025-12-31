

1) Colab nvcc and nvidia-smi dont match. These versions have to match for nsys and ncu files to be generated from nvidia profiling tools. 

2) pytorch and numpy broadcasting means the (3,) convention has to be converted to (3,1) and (3,1) has to be converted to (3,) using squeeze() and unsqueeze(). 


Notes: Attn)=_Impl and broadcasting and matrix mul need integration . MM starts with triangular for causal mask in attn. Start there and develop different attn archs for multimodal 


clean up NEMO install. installs on cuda but this is not useful. Proxy server with gcloud nvidial L4 better with clean nemo install. 


