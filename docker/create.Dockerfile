FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn

COPY c4_model.py model_sen.py create_model.py policy_head.py ./

CMD ["python3", "./create_model.py"]
