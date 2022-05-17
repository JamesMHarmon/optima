# Install Tensorflow, CUDA, and CUDNN Dependencies

## Clean all existing nvidia installs

https://medium.com/@anarmammadli/how-to-install-cuda-11-4-on-ubuntu-18-04-or-20-04-63f3dee2099

```console
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*

sudo rm -rf /usr/local/cuda*

sudo apt-get purge nvidia*

sudo apt-get update
sudo apt-get autoremove
sudo apt-get autoclean
```

## Install CUDA

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network

```console
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

### Set PATH for cuda 11.6 installation

Open ‘.profile’ file

```console
sudo nano ~/.profile
```

```bash
# set PATH for cuda 11.6 installation
if [ -d "/usr/local/cuda-11.6/bin/" ]; then
    export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

## Download CUDNN 8.4 for CUDA 11.*

### Download
https://developer.nvidia.com/rdp/cudnn-download

### Follow the installation guide

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

```console
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-*/7fa2af80.pub
sudo apt-get update
sudo apt-get install libcudnn8=8.4.0.27-1+cuda11.6
```

## Install Tensorflow

https://www.tensorflow.org/install/lang_c

```console
FILENAME=libtensorflow-gpu-linux-x86_64-2.7.0.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/libtensorflow/${FILENAME}
sudo tar -C /usr/local -xzf ${FILENAME}
```

### On Linux/macOS, if you extract the TensorFlow C library to a system directory, such as /usr/local, configure the linker with ldconfig:

```console
sudo ldconfig /usr/local/lib
```