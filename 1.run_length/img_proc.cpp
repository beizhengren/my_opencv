#include"img_proc.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

double convertDegree2Radian(double degree) {
    static constexpr double coefficient = CV_PI / 180.0;
    return degree * coefficient;
}

double convertRadian2Degree(double radian) {
    static constexpr double cofficient = 180.0 / CV_PI;
    return radian * cofficient;
}

void draw_box_with_points(cv::Mat& image, const cv::Point2f* const points, cv::Scalar color = { 0, 0, 255 }, int thickness = 2)
{
    for (int i = 0; i < 4; ++i)
    {
        cv::line(image, points[i], points[(i + 1) % 4], color, thickness);
    }
}

void draw_box_with_rect(cv::Mat& image, const cv::Rect& rect, cv::Scalar color = { 0, 255, 0 }, int thickness = 2)
{
    cv::rectangle(image, rect, color, thickness);
}

void draw_box_with_rotated_rect(cv::Mat& image, const cv::RotatedRect& rect, cv::Scalar color = { 0, 255, 0 }, int thickness = 2)
{
    cv::Point2f points[4];
    rect.points(points);
    draw_box_with_points(image, points, color, thickness);
}

int myOtsu(Mat& gray_img, int ignore_pixel, int gray_scale) {
    int otsu_thresh = 0;
    int total_pixels = 0;
    // n0 + n1 = total_pixels
    // n0
    int foreground_num = 0;
    // n1
    int background_num = 0;

    // w0 + w1 = 1;
    // w0 = n0 / total_pixels
    // = foreground_num / total_pixels;
    double foreground_ratio = 0.0;
    // w1 = n1 / total_pixels
    // = background_num / total_pixels
    double background_ratio = 0.0;

    // u = w0*u0 +w1*u1
    // u0 = sum_foreground_num_pixels / foreground_num
    double foreground_avg_pixel_value = 0.0;
    // u1 = sum_foreground_num_pixels / background_num
    double background_avg_pixel_value = 0.0;
    // u = w0 * u0 + w1 * u1 
    //   = foreground_ratio * foreground_avg_pixel_value + background_ratio * background_avg_pixel_value; 
    double total_avg_pixel_value = 0.0;
    // g = w0*w1*(u0 - u1)^2
    // = foreground_ratio * background_ratio * std::pow(foreground_avg_pixel_value - background_avg_pixel_value, 2);
    double inter_variance = 0.0;

    auto ratios = vector<float>(gray_scale, 0.0f);
    auto histogram = vector<int>(gray_scale, 0);

    // calculate the histogram
    for (size_t y = 0; y < gray_img.rows; ++y) {
        const uchar* cur_row_ptr = gray_img.ptr(y);
        for (size_t x = 0; x < gray_img.cols; ++x) {
            try {
                auto cur_pixel = cur_row_ptr[x];
                if (cur_pixel == ignore_pixel) { continue; }
                else {
                    ++histogram[cur_pixel];
                    ++total_pixels;
                }
            }
            catch (const std::exception& e) {
                cerr << e.what() << endl;
            }
        }
    }
    // calculate ratio of every pixel.
    for (size_t i = 0; i < gray_scale; ++i) {
        ratios[i] = histogram[i] / static_cast<float>(total_pixels);
    }

    for (size_t cur_thresh = 0; cur_thresh < gray_scale; ++cur_thresh) {
        double sum_foreground_pixel_values_with_ratio = 0;
        double sum_background_pixel_values_with_ratio = 0;
        double cur_inter_variance = 0.0;
        foreground_ratio = 0.0;
        background_ratio = 0.0;
        foreground_avg_pixel_value = 0.0;
        background_avg_pixel_value = 0.0;
        for (size_t idx = 0; idx < gray_scale; ++idx) {
            auto pixel_value = idx;
            if (pixel_value == ignore_pixel) { continue; }
            if (idx <= cur_thresh) { //foreground
                foreground_ratio += ratios[idx];
                sum_foreground_pixel_values_with_ratio += static_cast<double>(pixel_value) * ratios[idx];
            }
            else {//background
                background_ratio += ratios[idx];
                sum_background_pixel_values_with_ratio += static_cast<double>(pixel_value) * ratios[idx];
            }
        }
        foreground_avg_pixel_value = sum_foreground_pixel_values_with_ratio / foreground_ratio;
        background_avg_pixel_value = sum_background_pixel_values_with_ratio / background_ratio;
        cur_inter_variance = foreground_ratio * std::pow((foreground_avg_pixel_value - background_avg_pixel_value), 2) * background_ratio;
        if (cur_inter_variance > inter_variance) {
            inter_variance = cur_inter_variance;
            otsu_thresh = cur_thresh;
        }
    }
    return cvFloor(otsu_thresh);
}

Mat createBinaryImage(Mat& srcImage, Mat& mask) {

    resize(srcImage, srcImage, Size(500, 500), 0.0, 0.0, INTER_AREA);
#ifdef SHOW_IMAGE
    imshow("srcImage", srcImage);
#endif

    resize(mask, mask, Size(500, 500), 0.0, 0.0, INTER_AREA);
#ifdef SHOW_IMAGE
    imshow("maskImage", mask);
#endif

    Mat dstImage = Mat::zeros(srcImage.size(), srcImage.type());
    srcImage.copyTo(dstImage, mask);
    resize(dstImage, dstImage, Size(500, 500), 0.0, 0.0, INTER_AREA);
#ifdef SHOW_IMAGE
    imshow("dstImage", dstImage);
#endif

    Mat grayImage = dstImage.clone();
    cvtColor(dstImage, grayImage, CV_BGR2GRAY);
#ifdef SHOW_IMAGE
    imshow("grayImage", grayImage);
#endif

    int thresh = myOtsu(grayImage, 0);
    Mat binaryImage = grayImage.clone();
    cv::threshold(grayImage, binaryImage, thresh, 255, cv::THRESH_BINARY);
#ifdef SHOW_IMAGE
    imshow("binaryImage", binaryImage);
#endif

    return binaryImage;
}

