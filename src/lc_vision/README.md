# lc_vision

`lc_vision` now supports Astra RGB/depth capture, Foxglove H.264 debug streams, and optional person detection with runtime-selectable `openvino` or `ncnn` backends when they are available in the build.

## Install ncnn with vcpkg

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
~/vcpkg/vcpkg install "ncnn[vulkan]:arm64-linux" --recurse
```

## Export YOLO26 Models

### NCNN

```bash
cd /home/betty/help_ws/little_nav_car
./src/lc_vision/scripts/export_yolo26_ncnn.sh
```

The default NCNN model directory is:

```text
src/lc_vision/models/yolo26n_ncnn_model
```

### OpenVINO

```bash
cd /home/betty/help_ws/little_nav_car
./src/lc_vision/scripts/export_yolo26_openvino_fp16.sh
```

The default OpenVINO model directory is:

```text
src/lc_vision/models/yolo26n_openvino_model
```

You can switch to other exported Ultralytics models by overriding `detector.model_dir`. If `detector.model_dir` is empty, `lc_vision` chooses the default directory for the selected detector framework.

## Build

```bash
cd /home/betty/help_ws/little_nav_car
colcon build \
  --packages-select lc_vision \
  --cmake-args \
    -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DVCPKG_TARGET_TRIPLET=arm64-linux
```

## Runtime parameters

The detector is configured in `config/vision.yaml`:

- `detector.enable`
- `detector.framework`
- `detector.backend`
- `detector.model_dir`
- `detector.input_size`
- `detector.score_threshold`
- `detector.nms_threshold`
- `detector.fps`
- `detector.vulkan_device_index`
- `detector.cpu_num_threads`
- `detector.log_backend_info`
- `detector.target_classes`
- `depth.debug_video.histogram.enable`
- `depth.debug_video.histogram.topic`
- `depth.debug_video.peak_mask.enable`
- `depth.debug_video.peak_mask.topic`

`detector.framework` accepts `auto`, `openvino`, or `ncnn`.

`detector.backend` is interpreted by the selected framework:

- `openvino`: `auto`, `cpu`, `gpu`
- `ncnn`: `auto`, `vulkan`, `cpu`

When model loading fails, the node continues publishing RGB/depth debug video without detection overlays.

## Detection depth output

`lc_vision` now publishes `~/detections_depth` with per-box depth estimates derived from the depth ROI:

- invalid depths are removed with `depth.roi_min_valid_mm` / `depth.roi_max_valid_mm`
- valid depths are histogrammed with `depth.estimation.histogram_bin_size_mm`
- the dominant peak is expanded with `depth.estimation.peak_min_ratio` and `depth.estimation.peak_min_count`
- the final depth is the trimmed mean of the dominant peak samples using `depth.estimation.trim_ratio`

Useful runtime parameters:

- `depth.estimation.min_valid_pixels`
- `depth.estimation.min_peak_pixels`
- `depth.estimation.annotate_depth_on_rgb`
- `depth.estimation.annotate_depth_on_depth`
- `depth.debug_marker.enable`
- `depth.debug_marker.topic`
- `depth.debug_marker.frame_id`

When `depth.debug_marker.enable=true`, the node publishes a `visualization_msgs/MarkerArray` that draws circles centered at `base_link` with radius equal to the estimated depth in meters.

## Optional debug video streams

These streams are now runtime-configurable from `config/vision.yaml`. When enabled, the node can publish:

- `~/depth_histogram/video`
- `~/depth_peak_mask/video`
