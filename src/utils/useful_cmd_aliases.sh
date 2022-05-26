alias usboff="echo 0 | sudo tee /sys/bus/usb/devices/1-1.2/authorized && sudo uhubctl -a 0 -p 2 -l 1-1 -r 2"
alias usbon="sudo uhubctl -a 1 -p 2 -l 1-1 -r 2"

# https://newbedev.com/how-to-unlock-android-phone-through-adb
alias unlockandroid="adb shell input touchscreen swipe 930 880 930 380" 
alias startpytorchapp="adb shell am start -n org.pytorch.demo/.vision.ImageClassificationActivity"
alias stoppytorchapp="adb shell am force-stop org.pytorch.demo"
alias restoreoffline="usboff && adb kill-server && sleep 5 && usbon && adb devices && unlockandroid && stoppytorchapp"
alias testrpiimg="raspistill --awb sun -rot 180 -o cam.jpg"
alias syncandroidtimewithpi='adb root && adb shell "date `date +%m%d%H%M%G.%S`; am broadcast -a android.intent.action.TIME_SET" && adb unroot'

# usage:
# startrpicam /home/pi/Videos/elk-project/ 30 #30s vid to given dir
startrpicam() {
    # $1 -> store_directory $2 -> time_seconds
    raspivid -t 0 -fps 25 -b 1200000 -w 640 -h 480 -p 0,0,640,480 -vf -o - | ffmpeg -i - -vcodec copy -an -f mp4 -r 25 -t 50000 -pix_fmt yuv420p -use_wallclock_as_timestamps 1 -f segment -strftime 1 -segment_time $2 -segment_atclocktime 1 "$1"rpi_%Y-%m-%d_%H-%M-%S.mp4
}

# usage:
# startperceptcam ip /home/pi/Videos/elk-project/ 30 #30s vid to given dir
startperceptcam() {
	  # $1 -> percept_ip_address $2 -> store_directory $3 -> time_seconds
	  ffmpeg -rtsp_transport udp -i rtsp://ictd:ElkProject2021@"$1":8554/result -vcodec copy -an -f mp4 -r 25 -t 50000 -pix_fmt yuv420p -use_wallclock_as_timestamps 1 -f segment -strftime 1 -segment_time $3 -segment_atclocktime 1 "$2"percept_%Y-%m-%d_%H-%M-%S.mp4
}

# usage:
# startandroidcam /home/pi/Videos/elk-project/ 30
startandroidcam() {
   #android file android_YYYY_MM_DD_H_M.mp4
   unlockandroid && startpytorchapp && sleep $2 && stoppytorchapp && adb pull /storage/emulated/0/Android/data/org.pytorch.demo/files/android_*.mp4 $1
}

camdemo() {
  to_dir=/home/pi/Videos/elk-project/
  duration=30
  percept_ip=192.168.1.194
	startrpicam $to_dir $duration &
  startperceptcam $percept_ip $to_dir $duration &
  startandroidcam $to_dir $duration &
}

# usage:
# pullcamtrapdata /home/pi/Videos/elk-project/
pullcamtrapdata() {
   ct_home=/media/pi/STEALTHCAM/DCIM/100STLTH/
   for i in `ls $ct_home | head -n100`; do
	   f=`stat $ct_home/$i | grep Access | tail -1 | cut -d ' ' -f 2,3 | date -d - +%Y-%m-%d_%H-%M-%S.JPG`
	   mv $ct_home$i $2"percept_"$f
   done
}
