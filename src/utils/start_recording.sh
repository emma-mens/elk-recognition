#!/bin/bash
# use ffmpeg to continually record video files

# Adapted from audio in the reference to video: https://forums.raspberrypi.com/viewtopic.php?t=245288

# location to store audio files
todir="/home/pi/Videos/elk-project/"

# 5 minute .mp4 files
duration=300
percept_ip=192.168.1.194

# Record fixed-length MP4 files from rpi camera device continually
raspivid -t 0 -fps 25 -b 1200000 -w 640 -h 480 -p 0,0,640,480 -vf -o - | ffmpeg -i - -vcodec copy -an -f mp4 -r 25 -t 50000 -pix_fmt yuv420p -use_wallclock_as_timestamps 1 -f segment -strftime 1 -segment_time $duration -segment_atclocktime 1 "$todir"rpi_%Y-%m-%d_%H-%M-%S.mp4 &

sudo renice -n -20 $!

# Record fixed-length MP4 files from the Azure percept continually
ffmpeg -rtsp_transport udp -i rtsp://ictd:ElkProject2021@"$percept_ip":8554/result -vcodec copy -an -f mp4 -r 25 -t 50000 -pix_fmt yuv420p -use_wallclock_as_timestamps 1 -f segment -strftime 1 -segment_time $duration -segment_atclocktime 1 "$todir"percept_%Y-%m-%d_%H-%M-%S.mp4 &

sudo renice -n -20 $!

# Start recording from the phone
adb shell am start -n org.pytorch.demo/.vision.ImageClassificationActivity &


## TODO: check that the process is not already running for each of the commands above before running it
