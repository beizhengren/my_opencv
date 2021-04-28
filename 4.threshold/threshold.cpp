
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char** argv) {
    Mat src = imread(argv[1]);
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    // 全局二值化
    int th = 100;
    vector<cv::Mat> thresholds(7);
    cv::threshold(gray, thresholds[0], th, 255, THRESH_BINARY);
    cv::threshold(gray, thresholds[1], th, 255, THRESH_BINARY_INV);
    cv::threshold(gray, thresholds[2], th, 255, THRESH_TRUNC);
    cv::threshold(gray, thresholds[3], th, 255, THRESH_TOZERO);
    cv::threshold(gray, thresholds[4], th, 255, THRESH_TOZERO_INV);
    cv::threshold(gray, thresholds[5], th, 255, THRESH_OTSU);
    cv::threshold(gray, thresholds[6], th, 255, THRESH_TRIANGLE);
    // cv::threshold(gray, threshold[5], th, 255, THRESH_MASK);
    for (auto& threshold: thresholds) 
    {
        resize(threshold, threshold, cv::Size(300, 300), 0.0, 0.0, 1);
    }
    cv::imshow("THRESH_BINARY", thresholds[0]);
    cv::imshow("THRESH_BINARY_INV", thresholds[1]);
    cv::imshow("THRESH_TRUNC", thresholds[2]);
    cv::imshow("THRESH_TOZERO", thresholds[3]);
    cv::imshow("THRESH_TOZERO_INV", thresholds[4]);
    cv::imshow("THRESH_OTSU", thresholds[5]);
    cv::imshow("THRESH_TRIANGLE", thresholds[6]);
    //cv::imshow("THRESH_MASK", threshold6);
    cv::waitKey(0);

}


