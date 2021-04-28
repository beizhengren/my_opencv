#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

double convertDegree2Radian(double degree);

double convertRadian2Degree(double radian);

cv::Mat createBinaryImage(cv::Mat& srcImage, cv::Mat& mask);

int myOtsu(cv::Mat& gray_img, int ignore_pixel = -1, int gray_scale = 256);
