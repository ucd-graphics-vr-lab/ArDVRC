#pragma once
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>

#include "npy.hpp" // Reading/writing NPY files

class ArDVRCBase {

private:

protected:
    int N;
    bool blur;
    int blurKernel;
    int minCornerDistance;
    int radius;
    double cornerThreshold;
    cv::Mat footprint;

    std::vector<cv::Mat> kernels;
    std::vector<cv::Point2i> borderPoints;

public:
    // Constructor for ArDVRCBase
    ArDVRCBase(int N, bool blur, int blurKernel, int radius, int minDistance, double threshold);

    // Get/set N (N should only be 3 or 5!)
    int getN() { return N; }
    void setN(int n) { 
        assert(n == 3 || n == 5);
        N = n;
    }

    // Get/set ROI radius around a corner
    int getRadius() { return radius; }
    void setRadius(int r) { radius = r; }

    // ROI width and height will be radius*2+1
    int getROISize() { return radius * 2 + 1; }

    // Get/set blur kernel size
    int getBlurKernel() { return blurKernel; }
    void setBlurKernel(int b) { blurKernel = b; }

    // Get/set blur
    bool getBlur() { return blur; }
    void setBlur(bool b) { blur = b; }

    // Get/set minimum distance between corners
    int getMinCornerDistance() { return minCornerDistance; }
    void setMinCornerDistance(int m) { minCornerDistance = m; }

    // Get the x and y kernels
    void getXYKernels(int, cv::Mat&, cv::Mat&);

    // Get ROI border points, starting from top left corner, to top right, to bottom right, to bottom left, and then back to top left
    std::vector<cv::Point2i> getOrderedBorderPoints(int radius, int inner = 0);

    cv::Ptr<cv::aruco::Dictionary> getDictionaryFromFile(std::string _dictFile);
};
