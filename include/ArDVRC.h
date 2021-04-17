#pragma once
// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES
#define DllExport   __declspec( dllexport )

#include "ArDVRCBase.h"
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "picoflann.h"
//#include <utility>
#include <bitset>         // std::bitset

// PicoFlann for Point2f nearest neighbor searches
struct PicoFlann_Point2fAdapter {
    inline  float operator( )(const cv::Point& elem, int dim)const { return dim == 0 ? elem.x : elem.y; }
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
