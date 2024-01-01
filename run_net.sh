device_id=YOUR_DEVICE_ID
train_folder=YOUR_TRAIN_FOLDER
val_folder=YOUR_VAL_FOLDER

CUDA_VISIBLE_DEVICES=$device_id python run_net.py train --data_folder ${train_folder}
CUDA_VISIBLE_DEVICES=$device_id python run_net.py infer --data_folder ${val_folder}
