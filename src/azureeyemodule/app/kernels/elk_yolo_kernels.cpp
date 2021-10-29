// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "elk_yolo_kernels.hpp"

namespace cv {
namespace gapi {
namespace streaming {

std::tuple<GMat, GArray<float>> parse_yolov3(const GMat &in)
{
    return GParseYoloV3::on(in);
}

} // namespace streaming
} // namespace gapi
} // namespace cv
