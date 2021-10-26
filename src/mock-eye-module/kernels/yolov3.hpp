// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

// Standard libary includes
#include <tuple>
#include <vector>

// Third party includes
#include <opencv2/opencv.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace cv {
namespace gapi {
namespace custom {

// First version:
//
// Does nothing at all. Just passes the input through to the output.

// Op for parsing the output of the U-Net semantic segmentation network.
// This macro creates a G-API op. Remember, an op is the interface for a G-API "function",
// which can be written into a G-API graph.
//
// Since it is just the interface and not the implementation, we merely describe the input and the
// output of the operation here, while leaving the actual implementation up to the kernel we will
// write down below.
//
// For a little explanation: the macro takes three arguments:
// G_API_OP(op name, op function signature, op tag)
//
// The op name can be whatever you want. We have adopted the Intel convention of labeling ops
// GWhatever using camel case. Kernels are also camel case and follow the format GOCVWhatever.
// C++ wrapper functions use snake_case. Feel free to do whatever you want though.
//
// The function signature looks like this <output args(input args)>. Because C++ does not have native
// support for multiple return values (like Python), we need to wrap the multiple return values into
// a std::tuple if you have multiple outputs. Right now, we only have one output.
//
// A note about the types: in order to insert an op into a G-API graph, you need to make sure the types
// are mapped from what you actually want into the G-API type system.
// G-API types are simple: they have support for primatives, for cv::Mat objects as GMat objects,
// vectors as GArray objects, and everything else as GOpaque<type>.
//
// The tag can be whatever you want, and frankly, I'm not even sure what it's used for...
G_API_OP(GParseYoloV3, <GMat(GMat)>, "org.microsoft.gparseyolov3")
{
    // This boilerplate is required within curly braces following the macro.
    // You declare a static function called outMeta, which must take in WhateverDesc
    // versions of the types and output the same thing.
    //
    // Notice that we have mapped our inputs from GMat -> GMatDesc
    // (and added const and reference declaration syntax to the input).
    static GMatDesc outMeta(const GMatDesc&)
    {
        // This must be the right shape and type of the output. Otherwise you will get
        // an ominous error about the meta data being wrong or about a Mat object resizing in a kernel.
        return cv::GMatDesc(CV_32FC1, {1, 2535, 13});
    }
};

// This is the kernel declaration for the op.
// A single op can have several kernel implementations. That's kind of the whole point.
// The idea behind G-API is twofold: one is to make a declarative style computer vision
// pipeline syntax, and the other is to separate the declaration of the computer vision pipeline
// from its implementation. The reason for this is because you may want to have the same code
// that runs on a VPU, GPU, or CPU. If you are using G-API for it, all you would need to do
// is implement a GOCVParseUnetForSemSegCPU, GOCVParseUnetForSemSegGPU, and a GOCVParseUnetForSemSegVPU
// function. All the other code would remain the same.
//
// In our case, we are going to do everything in this function on the CPU, since there's not really
// any acceleration needed for this, our device cannot make use of the VPU for general programming,
// and because our VPU on the device is occupied running the neural network and doing a few other things anyway.
//
// So we will just create a single kernel, and it will run on the CPU.
GAPI_OCV_KERNEL(GOCVParseYoloV3, GParseYoloV3)
{
    // We need a static void run function.
    // It needs to take in the inputs as references and output the return values as references.
    //
    // So, since our op is of type <GMat(GMat)>
    // and since the kernel function needs to run good old fashioned C++ code (not G-API code),
    // we need to map this type to:
    //
    // <cv::Mat(cv::Mat)>
    //
    // but because we need to map our return value to an output reference,
    // the actual signature of this function is:
    // <void(const Mat&, Mat&)>
    static void run(const Mat &in_img, Mat &out_img)
    {
        // Here's where we will implement all the logic for post-processing our neural network
        // Our network outputs a shape {Batch Size, N Classes, Height, Width} tensor.
        CV_Assert(in_img.size.dims() == 3);
        CV_Assert(out_img.size.dims() == 3); // keep this as 3 for now since we're just copying
        CV_Assert(in_img.type() == out_img.type());
        CV_Assert(in_img.size == out_img.size);

        in_img.copyTo(out_img);

        // Note that we CANNOT do this, because it will throw an exception that the G-API detected that
        // a kernel parameter was reallocated.
        //out_img = in_img.clone();
    }
};

// Don't forget to declare the C++ wrapper function in our header!
GAPI_EXPORTS GMat parse_yolov3(const GMat &in);

} // namespace custom
} // namespace gapi
} // namespace cv
