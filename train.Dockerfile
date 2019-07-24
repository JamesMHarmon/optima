FROM tensorflow/tensorflow:1.13.2-gpu-py3-jupyter

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn

COPY c4_model.py model_sen.py train.py ./

# CMD ["python", "./train.py"]