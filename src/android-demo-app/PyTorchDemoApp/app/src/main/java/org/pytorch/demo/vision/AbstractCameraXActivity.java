package org.pytorch.demo.vision;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.widget.Toast;

import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.StatusBarUtils;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.VideoCapture;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.VideoCaptureConfig;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public abstract class AbstractCameraXActivity<R> extends BaseModuleActivity {
  private static final int REQUEST_CODE_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.WRITE_EXTERNAL_STORAGE};

  private long mLastAnalysisResultTime;
  @SuppressLint("SimpleDateFormat")
  private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH_mm");

  protected abstract int getContentViewLayoutId();

  protected abstract TextureView getCameraPreviewTextureView();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    StatusBarUtils.setStatusBarOverlay(getWindow(), true);
    setContentView(getContentViewLayoutId());

    startBackgroundThread();

    if (!allPermissionsGranted()) {
      ActivityCompat.requestPermissions(
          this,
          PERMISSIONS,
              REQUEST_CODE_PERMISSION);
    } else {
        setupCameraX();
    }
  }

  private Boolean allPermissionsGranted() {
    return ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED || grantResults[1] == PackageManager.PERMISSION_DENIED
      || grantResults[2] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
            this,
            "You can't use image classification example without granting CAMERA, AUDIO and WRITE_EXTERNAL_STORAGE permissions",
            Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        setupCameraX();
      }
    }
  }

  @SuppressLint("RestrictedApi")
  private void setupCameraX() {
    final TextureView textureView = getCameraPreviewTextureView();
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(output -> textureView.setSurfaceTexture(output.getSurfaceTexture()));

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(224, 224))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        (image, rotationDegrees) -> {
          if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) { // update frequency here
            return;
          }

          final R result = analyzeImage(image, rotationDegrees);
          if (result != null) {
            mLastAnalysisResultTime = SystemClock.elapsedRealtime();
            runOnUiThread(() -> applyToUiAnalyzeImageResult(result));
          }
        });

    final VideoCaptureConfig videoCaptureConfig = new VideoCaptureConfig.Builder().build();
    @SuppressLint("RestrictedApi") final VideoCapture videoCapture = new VideoCapture(videoCaptureConfig);

    setTimeoutStartCapture(videoCapture, 1000); // start recording 1s after app load

    CameraX.bindToLifecycle(this, imageAnalysis, videoCapture);
  }

  private void setTimeoutStartCapture(VideoCapture videoCapture, Integer timeout) {
    new android.os.Handler(Looper.getMainLooper()).postDelayed(
            () -> {
              startVideoRecording(videoCapture, getTimestampedFileName());
            }, timeout);
  }

  private void setTimeoutStopCapture(VideoCapture videoCapture, Integer timeout) {
    new android.os.Handler(Looper.getMainLooper()).postDelayed(
            () -> {
              stopVideoRecording(videoCapture);
            }, timeout);
  }

  @SuppressLint("RestrictedApi")
  private void stopVideoRecording(VideoCapture videoCapture) {
    // stop video recording
    videoCapture.stopRecording();
    Log.e("Debug video", " Stopping recording");
    setTimeoutStartCapture(videoCapture, 1000); // start recording again after 1s
  }

  @SuppressLint("RestrictedApi")
  private void startVideoRecording(VideoCapture videoCapture, File fileName) {
    Log.e("Debug video", " Recording to " + fileName);
    videoCapture.startRecording(fileName, new VideoCapture.OnVideoSavedListener() {

      @Override
      public void onVideoSaved(@NonNull File file) {
        Log.e("Debug video", " Video saved to " + file);
      }

      @Override
      public void onError(@NonNull VideoCapture.VideoCaptureError videoCaptureError,
                          @NonNull String message, @Nullable Throwable cause) {
        Log.e("Debug video", "Error saving video " + message);
      }
    });
    setTimeoutStopCapture(videoCapture, 5*60*1000); // stop recording after 5 minutes
  }

  private File getTimestampedFileName() {
    // filePath = /storage/emulated/0/Android/data/org.pytorch.demo/files/android_YYYY_MM_DD_H_M.mp4
    File file = new File(this.getExternalFilesDir(null),
            "/android_" + dateFormat.format(Calendar.getInstance().getTime()) + ".mp4");
    file.getParentFile().mkdirs();
    return file;
  }

  @WorkerThread
  @Nullable
  protected abstract R analyzeImage(ImageProxy image, int rotationDegrees);

  @UiThread
  protected abstract void applyToUiAnalyzeImageResult(R result);
}
