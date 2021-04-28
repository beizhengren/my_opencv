#include "img_proc.h"
#include "run_length.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat srcImage = imread(argv[1], IMREAD_COLOR);
    Mat mask = imread(argv[2], IMREAD_GRAYSCALE);
    Mat binaryImage = createBinaryImage(srcImage, mask);

    vector<vector<Point>>contours;
    vector<Vec4i> hierarchy;
    vector<RotatedRect> rotatedRects;

    erode(mask, mask, Mat());
    dilate(mask, mask, Mat());
    //Image should gray
    findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    auto runLengthPtr = std::make_unique<RunLength>(binaryImage);
    int areaThresh = 1000;
    //draw contour should one by one
    for (int i = 0; i < contours.size(); ++i) {
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        auto area = rotatedRect.size.area();
        cout << "contours " << i << ": ";
        if (area > areaThresh) {
            if (runLengthPtr->isPeriodic(contours[i])) {
                runLengthPtr->printInfo();
            }
            else {
                cout << "is not periodic" << endl;
            }
        }
        else {
            cout << "area is: " << area << " smaller than " << areaThresh << endl;
        }
#ifdef SHOW_IMAGE
        waitKey(0);
#endif
    }
}


