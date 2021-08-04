# Elk-recognition
Automatic elk detection on a farm.

# Tasks
Follow tasks [here](https://github.com/emma-mens/elk-recognition/issues/1)


# Youtube Data Download
To download the youtube data, use `data/sound_data.csv` along with the audiosetdl folder. Run the command

```
python download_audioset.py --data_dir /local1/emazuh/elk -b sound_data.csv -fp ../ffmpeg-git-20210611-amd64-static/ffprobe -f ../ffmpeg-git-20210611-amd64-static/ffmpeg 
```

# Training Yolov3

```
CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp4k8.txt --net_cfg=cfg/elk_yolov3.cfg --batch_size=24 --pretrained_weights=/local1/emazuh/elk/yolo/yolov3/weights/ --weights=/local1/emazuh/elk/yolo/yolov3/weights/acamp4k8 --epochs=400
```

# Training vggsound

