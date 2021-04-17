#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>




/**
 * @brief rotate the initial corner to get to the right position
 */
static void correctCornerPosition(std::vector< cv::Point2f >& _candidate, int rotate) {
    std::rotate(_candidate.begin(), _candidate.begin() + 4 - rotate, _candidate.end());
}

/**
  * @brief Given an input image and candidate corners, extract the bits of the candidate, including
  * the border bits
  */
static cv::Mat _extractBits(cv::InputArray _image, cv::InputArray _corners, int markerSize,
    int markerBorderBits, int cellSize, double cellMarginRate,
    double minStdDevOtsu) {

    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(_corners.total() == 4);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 1);
    CV_Assert(minStdDevOtsu >= 0);

    // number of bits in the marker
    int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
    int cellMarginPixels = int(cellMarginRate * cellSize);

    cv::Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    cv::Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr< cv::Point2f >(0)[0] = cv::Point2f(0, 0);
    resultImgCorners.ptr< cv::Point2f >(0)[1] = cv::Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr< cv::Point2f >(0)[2] =
        cv::Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr< cv::Point2f >(0)[3] = cv::Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    cv::Mat transformation = getPerspectiveTransform(_corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, cv::Size(resultImgSize, resultImgSize),
        cv::INTER_NEAREST);

    // output image containing the bits
    cv::Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, cv::Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all bits are the same color (black or white)
    cv::Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    cv::Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
        .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
    meanStdDev(innerRegion, mean, stddev);
    if (stddev.ptr< double >(0)[0] < minStdDevOtsu) {
        // all black or all white, depending on mean value
        if (mean.ptr< double >(0)[0] > 127)
            bits.setTo(1);
        else
            bits.setTo(0);
        return bits;
    }

    // now extract code, first threshold using Otsu
    threshold(resultImg, resultImg, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // for each cell
    for (int y = 0; y < markerSizeWithBorders; y++) {
        for (int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize)+cellMarginPixels;
            int Ystart = y * (cellSize)+cellMarginPixels;
            cv::Mat square = resultImg(cv::Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                cellSize - 2 * cellMarginPixels));
            // count white pixels on each cell to assign its value
            size_t nZ = (size_t)cv::countNonZero(square);
            if (nZ > square.total() / 2) bits.at< unsigned char >(y, x) = 1;
        }
    }

    return bits;
}

/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
static int _getBorderErrors(const cv::Mat& bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2 * borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

    int totalErrors = 0;
    for (int y = 0; y < sizeWithBorders; y++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr< unsigned char >(y)[k] != 0) totalErrors++;
            if (bits.ptr< unsigned char >(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for (int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for (int k = 0; k < borderSize; k++) {
            if (bits.ptr< unsigned char >(k)[x] != 0) totalErrors++;
            if (bits.ptr< unsigned char >(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}


/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
static uint8_t _identifyOneCandidate(const cv::Ptr<cv::aruco::Dictionary>& dictionary, cv::InputArray _image,
    std::vector<cv::Point2f>& _corners, int& idx,
    const cv::Ptr<cv::aruco::DetectorParameters>& params, int& rotation)
{
    CV_Assert(_corners.size() == 4);
    CV_Assert(_image.getMat().total() != 0);
    CV_Assert(params->markerBorderBits > 0);

    uint8_t typ = 1;
    // get bits
    cv::Mat candidateBits =
        _extractBits(_image, _corners, dictionary->markerSize, params->markerBorderBits,
            params->perspectiveRemovePixelPerCell,
            params->perspectiveRemoveIgnoredMarginPerCell, params->minOtsuStdDev);

    // analyze border bits
    int maximumErrorsInBorder =
        int(dictionary->markerSize * dictionary->markerSize * params->maxErroneousBitsInBorderRate);
    int borderErrors =
        _getBorderErrors(candidateBits, dictionary->markerSize, params->markerBorderBits);

    // check if it is a white marker
    if (params->detectInvertedMarker) {
        // to get from 255 to 1
        cv::Mat invertedImg = ~candidateBits - 254;
        int invBError = _getBorderErrors(invertedImg, dictionary->markerSize, params->markerBorderBits);
        // white marker
        if (invBError < borderErrors) {
            borderErrors = invBError;
            invertedImg.copyTo(candidateBits);
            typ = 2;
        }
    }
    if (borderErrors > maximumErrorsInBorder) return 0; // border is wrong

    // take only inner bits
    cv::Mat onlyBits =
        candidateBits.rowRange(params->markerBorderBits,
            candidateBits.rows - params->markerBorderBits)
        .colRange(params->markerBorderBits, candidateBits.rows - params->markerBorderBits);

    // try to indentify the marker
    if (!dictionary->identify(onlyBits, idx, rotation, params->errorCorrectionRate))
        return 0;

    return typ;
}
