# lc_vision

`lc_vision` now supports Astra RGB/depth capture, Foxglove H.264 debug streams, and optional person detection with `ncnn`.

## Install ncnn with vcpkg

```bash
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh
~/vcpkg/vcpkg install ncnn:arm64-linux
```

## Export YOLO26 to NCNN

```bash
cd /home/betty/help_ws/little_nav_car
./src/lc_vision/scripts/export_yolo26_ncnn.sh
```

The default model directory is:

```text
src/lc_vision/models/yolo26n_ncnn_model
```

You can switch to other exported Ultralytics NCNN models by overriding `detector.model_dir`.

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
- `detector.model_dir`
- `detector.input_size`
- `detector.score_threshold`
- `detector.nms_threshold`
- `detector.fps`
- `detector.target_classes`

When model loading fails, the node continues publishing RGB/depth debug video without detection overlays.
