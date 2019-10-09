FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn

COPY c4_model.py model_sen.py train_model.py ./

CMD ["python3", "./train_model.py"]