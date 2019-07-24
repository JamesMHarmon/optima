docker run --rm -it \
--runtime=nvidia \
# -p 8888:8888 \
--mount type=bind,source=$(pwd),target=/train \
-e SOURCE_MODEL_NAME=Connect4_run-1_00001 \
-e TARGET_MODEL_NAME=Connect4_run-1_00003 \
-e SOURCE_MODEL_PATH=/train/Connect4_runs/run-1/nets \
-e DATA_PATH=/train/training_data.json \
-e TRAIN_RATIO=0.9 \
-e TRAIN_BATCH_SIZE=512 \
-e EPOCHS=2 \
-e LEARNING_RATE=0.001 \
-e POLICY_LOSS_WEIGHT=1.0 \
-e VALUE_LOSS_WEIGHT=0.5 \
quoridor_engine/train:latest
