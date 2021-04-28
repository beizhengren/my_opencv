#include "img_proc.h"
#include "run_length.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>
#define SHOW_IMAGE true
using namespace cv;
using namespace std;

RunLength::RunLength(const Mat& binary_rect_image, const vector<Point>& contour) :
    contour_(contour), rotated_rect_(minAreaRect(contour)), radian_(0.0), degree_(0.0), cycle_(0), is_periodic_(false)
{
    binary_rect_image_ = binary_rect_image.clone();
    radians_ = { convertDegree2Radian(rotated_rect_.angle), convertDegree2Radian(rotated_rect_.angle - 90) };
}

RunLength::RunLength(const Mat& binary_rect_image) : radian_(0.0), degree_(0.0), cycle_(0), is_periodic_(false)
{
    binary_rect_image_ = binary_rect_image.clone();
}

bool RunLength::isPeriodic(const vector<Point>& contour) {
    clear();
    contour_.assign(contour.begin(), contour.end());
    rotated_rect_ = minAreaRect(contour_);
    radians_ = { convertDegree2Radian(rotated_rect_.angle), convertDegree2Radian(rotated_rect_.angle - 90) };
    is_periodic_ = isPeriodic();
    return is_periodic_;
}

bool RunLength::isPeriodic() {
    for (const auto& radian : radians_)
    {
        Vec2d current_direction = { cos(radian), sin(radian) };
        if (!getHistVec(current_direction)) {
            hist_vec_.clear();
            is_periodic_ = false;
            continue;
        };

        int cycle = calCycleSize();
        if (cycle > cycle_) {
            cycle_ = cycle;
            radian_ = radian;
            degree_ = convertRadian2Degree(radian_);
            direction_ = current_direction;
        }
        hist_vec_.clear();
    }

#ifdef SHOW_IMAGE
    Point2d start_point{ rotated_rect_.center.x - rotated_rect_.size.width / 2,  rotated_rect_.center.y + rotated_rect_.size.height / 2, };
    Point2d end_point = { start_point.x + binary_rect_image_.cols * direction_[0], start_point.y + binary_rect_image_.rows * direction_[1] };
    Mat test_image;
    cvtColor(binary_rect_image_, test_image, CV_GRAY2BGR);
    line(test_image, start_point, end_point, Scalar(255, 0, 0), 2);
    vector<vector<Point>> contours(1, contour_);
    drawContours(test_image, contours, 0, Scalar(255, 0, 0));

    imshow("test", test_image);
#endif
    if (cycle_ >= cycle_thresh) { is_periodic_ = true; }
    else { is_periodic_ = false; }
    return is_periodic_;
}

void RunLength::printInfo() {
    cout << "Contour Area is: " << contourArea(getContour()) << '\n'
        << "Cycle is: " << getCycle() << '\n'
        << "Degree is: " << getDegree() << '\n'
        << "Direction is " << getDirection() << '\n'
        << endl;
}

vector<Point> RunLength::getContour() {
    return contour_;
}

Vec2d RunLength::getDirection() {
    return direction_;
}

int RunLength::getCycle() {
    return cycle_;
}

double RunLength::getRadian() {
    return radian_;
}

double RunLength::getDegree() {
    return degree_;
}

int RunLength::calCycleSize()
{
    int threshold = getThreshold(hist_vec_);
    //cout << "threshold is " <<  threshold << endl;
    for (int i = 0; i < hist_vec_.size(); ++i) {
        if (hist_vec_[i] >= threshold) { hist_vec_[i] = 1; }
        else { hist_vec_[i] = 0; }
    }

    int left = 0;
    int right = 0;
    int cycle = 0;
    // this vector can be removed
    vector<int> width_vec;
    for (; right <= hist_vec_.size(); ++right) {
        if (right == hist_vec_.size() || hist_vec_[right] != hist_vec_[left]) {
            if (right - left > 1) {
                cycle++;
                width_vec.push_back(right - left);
            }
            left = right;
        }
    }
    return cycle;
}

int RunLength::getThreshold(vector<int >& hist_vec) {
    Mat mat(1, hist_vec.size(), CV_32FC1, hist_vec.data());
    int threshold = myOtsu(mat);
    return threshold;
}

bool RunLength::getHistVec(const Vec2d& current_direction) {
    struct MyCmp
    {
        bool operator()(const int& l, const int& r)const {
            return l < r;
        }
    };
    map<long long, int, MyCmp> hist_map;
    for (int y = 0; y < binary_rect_image_.rows; ++y) {
        const uchar* ptr = binary_rect_image_.ptr(y);
        for (int x = 0; x < binary_rect_image_.cols; ++x) {
            if (ptr[x] == 0 || pointPolygonTest(contour_, Point2f(x, y), false) == -1) { continue; }
            else {
                Vec2d current_vec = { static_cast<double>(x - 0.0), static_cast<double>(y - 0.0) };
                // distance from (0, 0) to current point
                double project_distance = current_vec.dot(current_direction);
                //double project_x = project_distance * current_direction[0];
                //double project_y = project_distance * current_direction[1];

                auto key = project_distance;
                auto pos = hist_map.find(key);

                if (pos == hist_map.cend()) {
                    //hist_map[key] = ptr[x];
                    hist_map[key] = 1;
                }
                else {
                    hist_map[key] += 1;
                    // 255 depends on image cols and rows and otsu's buckets size
                    if (hist_map[key] > 255) {
                        hist_map[key] = 255;
                    }
                }
            }
        }
    }
    if (hist_map.empty()) { return false; }
    else {
        for (auto& it : hist_map) {
            hist_vec_.push_back(it.second);
        }
        return true;
    }
}

void RunLength::clear() {
    contour_.clear();
    rotated_rect_ = RotatedRect();
    radians_.clear();
    hist_vec_.clear();
    direction_ = Vec2d(0, 0);

    radian_ = 0.0;
    degree_ = 0.0;
    cycle_ = 0;
    is_periodic_ = false;
}


