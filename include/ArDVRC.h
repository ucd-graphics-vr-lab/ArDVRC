#pragma once
// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES

//#include "ArDVRCBase.h"
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
/*#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>*/
#include <bitset>         // std::bitset

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


class ArDVRC : public ArDVRCBase {

private:
    cv::Mat input;

    cv::Mat response;
    cv::Mat peakMask;
    std::vector<cv::Mat> derivatives;
    std::vector<cv::Mat> convolutions;
    std::vector<cv::Mat> tensors;
    std::vector<cv::Point2i> corners;
    std::vector<cv::Point2i> rejectedCorners;
    cv::Point2f rho;
    std::vector<cv::Mat> imgRoi;
    std::vector<std::vector<cv::Mat>> convRois;
    std::vector<std::vector<cv::Point2i>> maxPoints;
    std::vector<std::vector<cv::Point3f>> lines;
    std::vector<cv::Point2f> intersections;
    std::vector<uint8_t> cornerLabels;
    std::vector<cv::Point2f> cornerDirections;
    std::vector<std::vector<cv::Point2i>> cornersByType;
    std::vector<std::vector<cv::Point2f>> cornerDirectionsByType;
    std::vector<std::vector<cv::Point2f>> templatePoints;
    std::vector<cv::Point2f> dstPoints;

    cv::Mat type1CornersMat;
    cv::Mat type1DirsMat;
    cv::Mat type1Dirs2DMat;
    cv::Mat type1Corners2DMat;
    cv::Mat directionIntersections, directionIntersectionsMask;
    std::vector<std::vector<cv::Point2f>> markerCornerLocations;// (dictionary->bytesList.rows);
    std::vector<std::vector<uint8_t>> markerCornerTypes;// (dictionary->bytesList.rows);
    std::vector<std::vector<cv::Point2f>> markerCornerDirections;// (dictionary->bytesList.rows);
    std::vector<std::vector<uint8_t>> markerCornerMasks;

    // Six different colors used for drawing
    std::vector<cv::Scalar> colors;

protected:
    // Image processing to compute corners/derivatives
    void computeDerivatives(const cv::Mat&, std::vector<cv::Mat>&);
    void computeConvolutions(const cv::Mat&, std::vector<cv::Mat>&, const std::vector<cv::Mat>&);
    void structureTensorSobel(const std::vector<cv::Mat>&, std::vector<cv::Mat>&);
    void cornerResponseFromTensors(const std::vector<cv::Mat>&, cv::Mat&, double);

    void getPeakMask(const cv::Mat&, cv::Mat&, double);
    std::vector<cv::Point2i> getHighIntensityPeaks(const cv::Mat&, const cv::Mat&, int);
    double getThreshold(const cv::Mat&, double);
    void ensureSpacingMatrix(const cv::Mat&, cv::Mat&, int);
    uint8_t classifyQuad(const std::vector<uint8_t>&);
    cv::Point2f intersection(const cv::Point3f&, const cv::Point3f&);
    cv::Point3f line(const cv::Point2f&, const cv::Point2f&);
    float vectorLength(const cv::Point2f&);
    uint8_t isAbove(const cv::Point2f&, const cv::Point2f&, const cv::Point2f&);
    uint8_t classifyCorner(const cv::Mat&, const std::vector<cv::Point2i>&, const cv::Point2f&, const std::vector<cv::Point2i>&, int);
    void getQuadMats(const std::vector<cv::Point2i>&, const cv::Point2f&, const std::vector<cv::Point2i>&, int, std::vector<cv::Mat>&);
    void excludeBorder(cv::Mat&, cv::Size&);
    void classifyCorners(int = 100, bool = false);
    void selectMaxXY(const std::vector<cv::Mat>&, std::vector<cv::Point2i>&);
    void getLines(const std::vector<cv::Mat>&, const std::vector<cv::Point2i>&, std::vector<cv::Point3f>&);
    void groupCornersByType();
    void createType1DirectionMat();
    void getRayIntersections(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&);
    void getUV(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&);
    void getSingularIntersections(const cv::Mat&, cv::Mat&);
    void adjustIntersectionsToCorners(std::vector<cv::Point2i>&, const cv::Mat&, int, int);
    void reorderCorners(std::vector<cv::Point2f>&);
    void getTemplate(int, int);
    void findHomographies(const std::vector<std::vector<cv::Point2f>>& _src, const std::vector<std::vector<cv::Point2f>>& _dst, std::vector<cv::Mat>& mats, std::vector<cv::Mat>& masks);
    void getPerspectiveTransforms(const std::vector<std::vector<cv::Point2f>>& _src, const std::vector<std::vector<cv::Point2f>>& _dst, std::vector<cv::Mat>& mats);
    int getFourCornersFromIntersections(std::vector<cv::Point2f>& _pts, int i, int j);
    int edgeLengthsAcceptable(std::vector<float> _magnitudes);

    cv::Mat getBitMat(int idNum);
    void getMarkerCornersFromBitMat(cv::Mat& _marker, std::vector<cv::Point2f>& _cornerLocations, std::vector<uint8_t>& _cornerTypes, std::vector<cv::Point2f>& _cornerDirections, std::vector<uint8_t>& _mask, int pixelsPerBit);

    std::vector<uint8_t> templateLabelsAll;

public:
    // Constructor/destructors
    ArDVRC(int, bool, int, int, int, std::string filename, double = 0.015f);

    /* Getters and setters */
    cv::Mat& getResponse() { return response; }
    std::vector<cv::Mat>& getConvolutions() {
        return convolutions;
    }
    cv::Mat& getPeakMask() { return peakMask; }
    std::vector<cv::Point2i>& getCorners() { return corners; }

    // Returns corner labels
    std::vector<uint8_t>& getCornerLabels() { return cornerLabels; }

    // Returns corner directions
    std::vector<cv::Point2f>& getCornerDirections() { return cornerDirections; }

    void processImage(const cv::Mat&);
    void drawCornerVectors(cv::Mat&, int = 10, bool = false, bool = false, bool = false);
    void drawDirectionVectors(cv::Mat& image, std::vector<cv::Point2f>& _corners, std::vector<cv::Point2f>& _directions, std::vector<uint8_t>& _types, int _arrowLength);
    int matchTemplate(std::vector<std::vector<cv::Point2f>>&, std::vector<int32_t>&);
    void detectMarkers(cv::Mat&, std::vector<std::vector<cv::Point2f>>&, std::vector<int>&);

    // Aruco
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> parameters;
};

/* GPU version of ArDVRC
TODO: NOT YET COMPLETE */
/*class ArDVRC_GPU : public ArDVRCBase {

private:
    cv::cuda::GpuMat tmp; // Temporary mat for processing
    cv::cuda::GpuMat input;
    cv::cuda::GpuMat response;
    cv::cuda::GpuMat peakMask;
    std::vector<cv::cuda::GpuMat> derivatives;
    std::vector<cv::cuda::GpuMat> convolutions;
    std::vector<cv::cuda::GpuMat> tensors;
    cv::Ptr<cv::cuda::Filter> boxFilter;
    cv::Ptr<cv::cuda::Filter> dilateFilter;
    std::vector<cv::Ptr<cv::cuda::Filter>> sobelFilters;
    std::vector<cv::Ptr<cv::cuda::Filter>> gaussianFilters;
    std::vector<cv::Ptr<cv::cuda::Filter>> convolutionFilters;

public:
    ArDVRC_GPU(int, bool, int, int, int, double = 0.015f);

    cv::cuda::GpuMat& getResponse() { return response; }
    std::vector<cv::cuda::GpuMat>& getConvolutions() { return convolutions; }
    cv::cuda::GpuMat& getPeakMask() { return peakMask; }

    void processImage(cv::Mat&);
    void computeDerivatives(cv::cuda::GpuMat&, std::vector<cv::cuda::GpuMat>&);
    void computeConvolutions(cv::cuda::GpuMat&, std::vector<cv::cuda::GpuMat>&);
    void structureTensorSobel(std::vector<cv::cuda::GpuMat>&, std::vector<cv::cuda::GpuMat>&);
    void cornerResponseFromTensors(std::vector<cv::cuda::GpuMat>&, cv::cuda::GpuMat&, double);
    void _getPeakMask(cv::cuda::GpuMat&, cv::cuda::GpuMat&, double);
    double _getThreshold(cv::cuda::GpuMat&, double);
};

*/