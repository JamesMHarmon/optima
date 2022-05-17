FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /usr/local/bin

RUN pip3 install keras sklearn pandas

COPY c4_model.py model_sen.py train_model.py  warmup_lr_scheduler.py tensorboard_enriched.py fit.py split_file_data_generator.py metric.py ./

CMD ["python3", "./train_model.py"]