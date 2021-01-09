#include <algorithm>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

constexpr auto kWindowName = "BRISK BF Matcher";
constexpr auto kThresholdTaskbar = "Threshold";
constexpr auto kOctavesTaskbar = "Octaves";
constexpr auto kPatternScaleTaskbar = "Pattern Scale";

int main(int argc, char* argv[]) {
  if (argc <= 2) {
    std::cout << "Format to call: " << argv[0] << " template_image scene_image"
              << '\n';
    return EXIT_FAILURE;
  }

  const std::string template_image_filename = argv[1];
  const std::string scene_image_filename = argv[2];

  auto template_img = cv::imread(template_image_filename);

  if (template_img.empty()) {
    std::cout << "Error reading: " << template_image_filename << '\n';
    return EXIT_FAILURE;
  }

  auto scene_image = cv::imread(scene_image_filename);

  if (scene_image.empty()) {
    std::cout << "Error reading: " << scene_image_filename << '\n';
    return EXIT_FAILURE;
  }

  cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

  // Detector parameters
  int current_threshold = 30;
  int current_octaves = 3;
  int current_scale = 3;

  // Detector parameters modifiers
  cv::createTrackbar(kThresholdTaskbar, kWindowName, &current_threshold, 128);
  cv::createTrackbar(kOctavesTaskbar, kWindowName, &current_octaves, 9);
  cv::createTrackbar(kPatternScaleTaskbar, kWindowName, &current_scale, 9);

  while (true) {
    // Update detector parameters
    current_threshold = cv::getTrackbarPos(kThresholdTaskbar, kWindowName);
    current_octaves = cv::getTrackbarPos(kOctavesTaskbar, kWindowName);
    current_scale =
        (cv::getTrackbarPos(kPatternScaleTaskbar, kWindowName) + 1) / 3;

    auto detector = cv::KAZE::create();
    if (!detector) {
      std::cout << "Error creating feature detector\n";
      return EXIT_FAILURE;
    }

    // Detect keypoints
    std::vector<cv::KeyPoint> kp_template;
    detector->detect(template_img, kp_template);

    std::vector<cv::KeyPoint> kp_scene;
    detector->detect(scene_image, kp_scene);

    auto descriptor =
        cv::BRISK::create(current_threshold, current_octaves, current_scale);
    if (!detector) {
      std::cout << "Error creating feature descriptor\n";
      return EXIT_FAILURE;
    }

    // Calculate descriptors
    cv::Mat des_template;
    descriptor->compute(template_img, kp_template, des_template);

    cv::Mat des_scene;
    descriptor->compute(scene_image, kp_scene, des_scene);

    // Create matcher
    // algorithm=FLANN_INDEX_LSH,  table_number=6,
    // key_size=12, multi_probe_level=1
    auto index_params = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
    if (!index_params) {
      std::cout << "Error creating feature matcher index params\n";
      return EXIT_FAILURE;
    }

    // Then set number of searches. Higher is better, but takes longer
    auto search_params = cv::makePtr<cv::flann::SearchParams>(100);
    if (!search_params) {
      std::cout << "Error creating feature matcher search params\n";
      return EXIT_FAILURE;
    }

    cv::FlannBasedMatcher matcher(index_params, search_params);

    // Match features
    std::vector<cv::DMatch> matches;
    matcher.match(des_template, des_scene, matches);

    // Sort them in the order of their distance.
    std::sort(matches.begin(), matches.end());

    // Use the 10 best features only
    matches.resize(std::min<int>(10, matches.size()));

    // Draw matches
    cv::Mat out_img;
    cv::drawMatches(template_img, kp_template, scene_image, kp_scene, matches,
                    out_img, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow(kWindowName, out_img);

    // Calculate homography
    // Consider point filtering
    std::vector<cv::Point2f> template_points;
    std::vector<cv::Point2f> scene_points;
    for (const auto& match : matches) {
      template_points.push_back(kp_template[match.queryIdx].pt);
      scene_points.push_back(kp_scene[match.trainIdx].pt);
    }

    // Caclulate homography
    cv::Mat H = cv::findHomography(template_points, scene_points, cv::RANSAC);

    // Draw projection
    if (!H.empty()) {
      // Template image corners
      const std::vector<cv::Point2f> template_corners{
          {0.0f, 0.0f},
          {1.0f * template_img.cols - 1, 0.0f},
          {1.0f * template_img.cols - 1, 1.0f * template_img.rows - 1},
          {0.0f, 1.0f * template_img.rows - 1},
          {0.0f, 0.0f},
      };

      // Transform template image corners to scene
      std::vector<cv::Point2f> warped_points;
      cv::perspectiveTransform(template_corners, warped_points, H);

      auto warped_image = scene_image.clone();

      for (int i = 0; i < template_corners.size() - 1; i++) {
        cv::line(warped_image, warped_points[i], warped_points[i + 1],
                 cv::Scalar(0, 255, 0));
      }

      cv::namedWindow("Warped Object", cv::WINDOW_AUTOSIZE);
      cv::imshow("Warped Object", warped_image);
    }

    // Press ESC to leave
    if (cv::waitKey(10) == 27) {
      break;
    }
  }

  return EXIT_SUCCESS;
}
