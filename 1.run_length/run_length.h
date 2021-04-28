#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>
#define SHOW_IMAGE true
   
class RunLength {
public:
    struct Direction {
        Direction(float dir_x, float dir_y) :vx(dir_x), vy(dir_y) {}
        Direction& operator=(const Direction& direction)
        {
            vx = direction.vx;
            vy = direction.vy;
        }
        float vx;
        float vy;
    };

    RunLength() = delete;
    explicit RunLength(const cv::Mat& binary_rect_image, const std::vector<cv::Point>& contour);
    explicit RunLength(const cv::Mat& binary_rect_image);

    bool isPeriodic(const std::vector<cv::Point>& contour);
    bool isPeriodic();

    void printInfo();

    std::vector<cv::Point> getContour();

    cv::Vec2d getDirection();

    int getCycle();

    double getRadian();

    double getDegree();

private:
    int calCycleSize();

    int getThreshold(std::vector<int >& hist_vec);

    bool getHistVec(const cv::Vec2d& current_direction);

    void clear();
private:
    cv::Mat binary_rect_image_;

    std::vector<cv::Point> contour_;
    cv::RotatedRect rotated_rect_;
    std::vector<double> radians_;
    std::vector<int> hist_vec_;
    cv::Vec2d direction_;

    double radian_;
    double degree_;
    int cycle_;
    bool is_periodic_;

    static constexpr int cycle_thresh = 3;
};


