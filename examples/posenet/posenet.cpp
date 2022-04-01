/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoOutput.h"
#include "videoSource.h"

#include "poseNet.h"

#include <signal.h>

bool signal_recieved = false;

void sig_handler(int signo) {
    if (signo == SIGINT) {
        LogVerbose("received SIGINT\n");
        signal_recieved = true;
    }
}

int usage() {
    printf("usage: posenet [--help] [--network=NETWORK] ...\n");
    printf("                input_URI [output_URI]\n\n");
    printf("Run pose estimation DNN on a video/image stream.\n");
    printf("See below for additional arguments that may not be shown above.\n\n");
    printf("positional arguments:\n");
    printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
    printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

    printf("%s", poseNet::Usage());
    printf("%s", videoSource::Usage());
    printf("%s", videoOutput::Usage());
    printf("%s", Log::Usage());

    return 0;
}

int main(int argc, char **argv) {
    /*
     * parse command line
     */
    commandLine cmdLine(argc, argv);

    if (cmdLine.GetFlag("help"))
        return usage();

    /*
     * attach signal handler
     */
    if (signal(SIGINT, sig_handler) == SIG_ERR)
        LogError("can't catch SIGINT\n");

    /*
     * create input stream
     */
    videoSource *input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if (!input) {
        LogError("posenet: failed to create input stream\n");
        return 0;
    }

    /*
     * create output stream
     */
    videoOutput *output = videoOutput::Create(cmdLine, ARG_POSITION(1));

    if (!output)
        LogError("posenet: failed to create output stream\n");

    /*
     * create recognition network
     */
    poseNet *net = poseNet::Create(cmdLine);

    if (!net) {
        LogError("posenet: failed to initialize poseNet\n");
        return 0;
    }

    // parse overlay flags
    const uint32_t overlayFlags =
        poseNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "links,keypoints"));

    /*
     * processing loop
     */
    while (!signal_recieved) {
        // capture next image image
        uchar3 *image = NULL;

        if (!input->Capture(&image, 1000)) {
            // check for EOS
            if (!input->IsStreaming())
                break;

            LogError("posenet: failed to capture next frame\n");
            continue;
        }

        // run pose estimation
        std::vector<poseNet::ObjectPose> poses;

        if (!net->Process(image, input->GetWidth(), input->GetHeight(), poses, overlayFlags)) {
            LogError("posenet: failed to process frame\n");
            continue;
        }

        LogInfo("posenet: detected %zu %s(s)\n", poses.size(), net->GetCategory());

        // render outputs
        if (output != NULL) {
            output->Render(image, input->GetWidth(), input->GetHeight());

            // update status bar
            char str[256];
            sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR,
                    NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()),
                    net->GetNetworkFPS());
            output->SetStatus(str);

            // check if the user quit
            if (!output->IsStreaming())
                signal_recieved = true;
        }

        // print out timing info
        net->PrintProfilerTimes();
    }

    /*
     * destroy resources
     */
    LogVerbose("posenet: shutting down...\n");

    SAFE_DELETE(input);
    SAFE_DELETE(output);
    SAFE_DELETE(net);

    LogVerbose("posenet: shutdown complete.\n");
    return 0;
}
