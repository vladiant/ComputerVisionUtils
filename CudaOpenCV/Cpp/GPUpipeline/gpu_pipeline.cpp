#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

// https://learnopencv.com/getting-started-opencv-cuda-modul/

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Program usage: " << argv[0] << " video_filename\n";
    return -1;
  }

  // init video capture with video
  cv::VideoCapture capture(argv[1]);
  if (!capture.isOpened()) {
    // error in opening the video file
    std::cout << "Unable to open file!" << std::endl;
    return -1;
  }

  // get default video FPS
  double fps = capture.get(cv::CAP_PROP_FPS);

  // get total number of video frames
  int num_frames = int(capture.get(cv::CAP_PROP_FRAME_COUNT));

  std::cout << "Number of frames: " << num_frames << std::endl;

  // read the first frame
  cv::Mat frame, previous_frame;
  capture >> frame;

  // resize frame
  cv::resize(frame, frame, cv::Size(960, 540), 0, 0, cv::INTER_LINEAR);

  // convert to gray
  cv::cvtColor(frame, previous_frame, cv::COLOR_BGR2GRAY);

  // upload pre-processed frame to GPU
  cv::cuda::GpuMat gpu_previous;
  gpu_previous.upload(previous_frame);

  // declare cpu outputs for optical flow
  cv::Mat hsv[3], angle, bgr;

  // declare gpu outputs for optical flow
  cv::cuda::GpuMat gpu_magnitude, gpu_normalized_magnitude, gpu_angle;
  cv::cuda::GpuMat gpu_hsv[3], gpu_merged_hsv, gpu_hsv_8u, gpu_bgr;

  // set saturation to 1
  hsv[1] = cv::Mat::ones(frame.size(), CV_32F);
  gpu_hsv[1].upload(hsv[1]);

  std::map<std::string, std::vector<float>> timers;

  while (true) {
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    // start full pipeline timer
    auto start_full_time = high_resolution_clock::now();

    // start reading timer
    auto start_read_time = high_resolution_clock::now();

    // capture frame-by-frame
    capture >> frame;

    if (frame.empty()) break;

    // upload frame to GPU
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);

    // end reading timer
    auto end_read_time = high_resolution_clock::now();

    // add elapsed iteration time
    timers["reading"].push_back(
        duration_cast<milliseconds>(end_read_time - start_read_time).count() /
        1000.0);

    // start pre-process timer
    auto start_pre_time = high_resolution_clock::now();

    // resize frame
    cv::cuda::resize(gpu_frame, gpu_frame, cv::Size(960, 540), 0, 0,
                     cv::INTER_LINEAR);

    // convert to gray
    cv::cuda::GpuMat gpu_current;
    cv::cuda::cvtColor(gpu_frame, gpu_current, cv::COLOR_BGR2GRAY);

    // end pre-process timer
    auto end_pre_time = high_resolution_clock::now();

    // add elapsed iteration time
    timers["pre-process"].push_back(
        duration_cast<milliseconds>(end_pre_time - start_pre_time).count() /
        1000.0);

    // start optical flow timer
    auto start_of_time = high_resolution_clock::now();

    // create optical flow instance
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> ptr_calc =
        cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 15, 3, 5, 1.2, 0);
    // calculate optical flow
    cv::cuda::GpuMat gpu_flow;
    ptr_calc->calc(gpu_previous, gpu_current, gpu_flow);

    // end optical flow timer
    auto end_of_time = high_resolution_clock::now();

    // add elapsed iteration time
    timers["optical flow"].push_back(
        duration_cast<milliseconds>(end_of_time - start_of_time).count() /
        1000.0);

    // start post-process timer
    auto start_post_time = high_resolution_clock::now();

    // split the output flow into 2 vectors
    cv::cuda::GpuMat gpu_flow_xy[2];
    cv::cuda::split(gpu_flow, gpu_flow_xy);

    // convert from cartesian to polar coordinates
    cv::cuda::cartToPolar(gpu_flow_xy[0], gpu_flow_xy[1], gpu_magnitude,
                          gpu_angle, true);

    // normalize magnitude from 0 to 1
    cv::cuda::normalize(gpu_magnitude, gpu_normalized_magnitude, 0.0, 1.0,
                        cv::NORM_MINMAX, -1);

    // get angle of optical flow
    gpu_angle.download(angle);
    angle *= ((1 / 360.0) * (180 / 255.0));

    // build hsv image
    gpu_hsv[0].upload(angle);
    gpu_hsv[2] = gpu_normalized_magnitude;
    cv::cuda::merge(gpu_hsv, 3, gpu_merged_hsv);

    // multiply each pixel value to 255
    gpu_merged_hsv.cv::cuda::GpuMat::convertTo(gpu_hsv_8u, CV_8U, 255.0);

    // convert hsv to bgr
    cv::cuda::cvtColor(gpu_hsv_8u, gpu_bgr, cv::COLOR_HSV2BGR);

    // send original frame from GPU back to CPU
    gpu_frame.download(frame);

    // send result from GPU back to CPU
    gpu_bgr.download(bgr);

    // update previous_frame value
    gpu_previous = gpu_current;

    // end post pipeline timer
    auto end_post_time = high_resolution_clock::now();

    // add elapsed iteration time
    timers["post-process"].push_back(
        duration_cast<milliseconds>(end_post_time - start_post_time).count() /
        1000.0);

    // end full pipeline timer
    auto end_full_time = high_resolution_clock::now();

    // add elapsed iteration time
    timers["full pipeline"].push_back(
        duration_cast<milliseconds>(end_full_time - start_full_time).count() /
        1000.0);

    // visualization
    cv::imshow("original", frame);
    cv::imshow("result", bgr);
    int keyboard = cv::waitKey(1);
    if (keyboard == 27) break;
  }

  // elapsed time at each stage
  std::cout << "Elapsed time" << std::endl;
  for (auto const& timer : timers) {
    std::cout << "- " << timer.first << " : "
              << accumulate(timer.second.begin(), timer.second.end(), 0.0)
              << " seconds" << std::endl;
  }

  // calculate frames per second
  std::cout << "Default video FPS : " << fps << std::endl;
  float optical_flow_fps =
      (num_frames - 1) / std::accumulate(timers["optical flow"].begin(),
                                         timers["optical flow"].end(), 0.0);
  std::cout << "Optical flow FPS : " << optical_flow_fps << std::endl;

  float full_pipeline_fps =
      (num_frames - 1) / std::accumulate(timers["full pipeline"].begin(),
                                         timers["full pipeline"].end(), 0.0);
  std::cout << "Full pipeline FPS : " << full_pipeline_fps << std::endl;

  return 0;
}
