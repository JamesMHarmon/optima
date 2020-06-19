FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /usr/local/bin

RUN pip3 install 'keras<2.4.0' sklearn

COPY c4_model.py model_sen.py create_model.py warmup_lr_scheduler.py get_gradient_norm.py tensorboard_enriched.py ./

CMD ["python3", "./create_model.py"]
