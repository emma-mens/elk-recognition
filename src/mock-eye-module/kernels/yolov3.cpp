// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "yolov3.hpp"

namespace cv {
namespace gapi {
namespace custom {

// First version:
//
// This is pretty much pure boilerplate and is not strictly necessary.
// We just wrap the invocation of our op with a traditional C++ function
// so that we can just call parse_unet_for_semseg() rather than use
// the stranger looking GParseUnetForSemSeg::on() syntax.
// Really, that's all.
GMat parse_yolov3(const GMat &in)
{
    return GParseYoloV3::on(in);
}

} // namespace custom
} // namespace gapi
} // namespace cv
