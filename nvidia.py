import torch
import os
import time
from functools import wraps
def hello():
	print("hello")


def timeit(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
    return result
  return wrapper

deb_file = '/content/drive/MyDrive/cuda-keyring_1.1-1_all.deb'

@timeit
def update_12_4():
	print("colab_dc333.update_12_4 downgrades colab from cuda12.5 to cuda12.4 to make nvcc and nvidia-smi match")
	print("this takes 5minutes, please wait ...")
	os.environ['DEBIAN_FRONTEND'] = "noninteractive"
	os.system('apt-get update && apt-get upgrade -y')
	os.system('apt-get install -y git emacs keyboard-configuration')

	if os.path.exists(deb_file):
		os.system('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb')
		os.system('apt-get -y install cuda-toolkit-12-4')
	else:
		os.system('dpkg -i /content/drive/MyDrive/cuda-keyring_1.1-1_all.deb')
		os.system('apt-get -y install cuda-toolkit-12-4')
	#os.system('apt-get update')
	#fix links
	os.system('rm /etc/alternatives/cuda')
	os.system('ln -s  /usr/local/cuda-12.4 /etc/alternatives/cuda')
	os.system('nvcc --version')

def install_utils():
	os.system('apt-get install -y git net-tools mlocate keyboard-configuration')

@timeit
def apt_fast():
	print("colab_dc333.test_apt_fast uses apt-fast vs apt and downgrades colab from cuda12.5 to cuda12.4 to make nvcc and nvidia-smi match")
	print("this takes unknown minutes, please wait ...")
	os.environ['DEBIAN_FRONTEND'] = "noninteractive"
	os.system('apt-fast add-apt-repository -y ppa:apt-fast/stable')
	os.system('apt-get update')
	os.system('apt-get install -y apt-fast')
	os.system('apt-fast update && apt-fast upgrade -y')
	os.system('apt-fast install emacs git net-tools mlocate keyboard-configuration')
	if os.path.exists(deb_file):
		os.system('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb')
		os.system('apt-fast -y install cuda-toolkit-12-4')
	else:
		os.system('dpkg -i /content/drive/MyDrive/cuda-keyring_1.1-1_all.deb')
		os.system('apt-fast -y install cuda-toolkit-12-4')
	os.system('rm /etc/alternatives/cuda')
	os.system('ln -s  /usr/local/cuda-12.4 /etc/alternatives/cuda')
	os.system('nvcc --version')
	
@timeit
def install_nemo():
	os.system('pip install --no-build-isolation nemo_toolkit["all"]')
	os.system('pip install --no-build-isolation megatron-core')
	os.system('pip install --no-build-isolation transformer-engine[pytorch]')
	os.system('pip install --disable-pip-version-check --no-build-isolation --no-cache-dir git+https://github.com/state-spaces/mamba.git')
	os.system('pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir  git+https://github.com/NVIDIA/apex.git')
	os.system('pip install git+https://github.com/NVIDIA/NeMo-Run.git')
	print("test install with >>from nemo.collections import llm")

def install_nsys():
	os.system('apt update')
	os.system('apt install -y --no-install-recommends gnupg')
	os.system('echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list')
	os.system('apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub')
	os.system('apt update')
	os.system('apt install nsight-systems-cli')
	
@timeit
def install_litgpt():
	os.system('pip install lit-gpt[all]')
	os.system('pip install datasets')
	os.system('pip install hf_transfer')
	os.system('pip install litdata')

def device():
	device = torch.device('cpu')
	if torch.cuda.is_available():
		device = torch.device('cuda')
		gpu_stats = torch.cuda.get_device_properties(0)
		start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
		max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
		print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
		print(f"{start_gpu_memory} GB of memory reserved.")
	torch.set_default_device(device)
	print(f"Using device = {torch.get_default_device()}")

def gpu_memory():
	gpu_stats = torch.cuda.get_device_properties(0)
	start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
	max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
	print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
	print(f"{start_gpu_memory} GB of memory reserved.")


