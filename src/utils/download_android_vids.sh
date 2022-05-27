ANDROID_VID_DIR=/storage/emulated/0/Android/data/org.pytorch.demo/files/
ELK_VID_DIR=/home/pi/Videos/elk-project/
for f in `adb shell ls $ANDROID_VID_DIR | head -n100`
do
  echo "$ANDROID_VID_DIR"$f
  adb pull "$ANDROID_VID_DIR"$f $ELK_VID_DIR && sleep 1 && adb shell rm "$ANDROID_VID_DIR"$f
  sleep 1
done
