
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main() {

    Mat img(500, 500,  CV_8UC3,Scalar(0, 0, 0));
    imshow("test_image", img);
    waitKey(-1);
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    vector<Point> contour;
    contour.push_back(Point{ 0, 0 });
    contour.push_back(Point{ 100,0 });
    contour.push_back(Point{ 100,100 });
    contour.push_back(Point{ 0, 100 });
    contour.push_back(Point{ 200, 100 });
    contour.push_back(Point{ 50, 50 });
    contour.push_back(Point{ 50,200 });
    contours.push_back(contour);

    vector<vector<Point>>hull(contours.size());
    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        convexHull(Mat(contours[i]), hull[i], false);
    }

    for (int idx = 0; idx < contours.size(); ++idx)
    {
        drawContours(img, hull, idx, Scalar(255, 0, 0), CV_FILLED, 8, hierarchy);
        imshow("test_image", img);
        waitKey(-1);
    }
    // draw minAreaRect
    auto rect = minAreaRect(contours[0]);
    vector<Point2f> points(4);
    rect.points(points.data());
    for (int i = 0; i < 4; ++i)
    {
        line(img, points[i], points[(i + 1) % 4], Scalar(0,0,255), 5);
    }
    imshow("test_image", img);
    waitKey(-1);
}

