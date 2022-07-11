FROM tensorflow/tensorflow:2.9.1-gpu-jupyter

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn pandas pyhocon

COPY model_py /tf
