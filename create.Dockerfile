FROM tensorflow/tensorflow:1.13.2-gpu-py3

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn

COPY c4_model.py model_sen.py create_model.py ./

CMD ["python3", "./create_model.py"]
