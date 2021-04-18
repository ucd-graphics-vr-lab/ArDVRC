// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <numeric>
#include <Windows.h>
#include <cnpy.h>
#include <limits>
#include "../../../include/ArDVRC.h"

int main()
{
    // Timers
    std::chrono::steady_clock::time_point begin, end;

    // Image scaling factor
    const float SCALE = 0.5f;//0.5f normally
    const bool BLUR = true;
    const int BLUR_KERNEL = 3;
    const int RADIUS = 4;
    const int ROI_SHAPE = RADIUS * 2 + 1;
    const int N = 5;

    ArDVRC ardvrc(N, BLUR, BLUR_KERNEL, RADIUS, RADIUS, "..\\..\\..\\data\\entropy_dictionary\\DICT_4X4_64_ENTROPY.npy", 0.00015f);

    cv::Mat img, imgColor;

    // Set this to however many images are in the dataset
    int nImages = 10000;

    // Storage for data
    // Ids found for each frame
    std::vector<int> arucoIds;
    std::vector<int> ardvrcIds;

    // Iteration when ID was detected
    std::vector<int> arucoFrames;
    std::vector<int> ardvrcFrames;

    // Corners of the markers
    std::vector<int> arucoCorners;
    std::vector<int> ardvrcCorners;

    // Timing information
    std::vector<float> arucoTime(nImages);
    std::vector<float> ardvrcTime(nImages);

    // Path to the datasets (assuming you've already generated some datasets)
    std::string path = "..\\..\\..\\data\\data_generation\\datasets\\";

    // Data folder. Make sure this folder exists and contains TIFF images    
    std::string folder = "control\\";
    //std::string folder = "occluding_edge_ellipse\\";
    //std::string folder = "noise\\";
    //std::string folder = "background_shapes\\";
    //std::string folder = "motion_blur_horizontal_kernel_10\\";
    //std::string folder = "motion_blur_horizontal_kernel_20\\";
    //std::string folder = "motion_blur_horizontal_kernel_30\\";
    //std::string folder = "decreased_lighting\\";

    std::string outfolder = path + folder + "results\\";

    // Creates a local results folder for the output data
    std::wstring stemp = std::wstring(outfolder.begin(), outfolder.end());
    LPCWSTR folderL = stemp.c_str();
    if (CreateDirectory(folderL, NULL) ||
        ERROR_ALREADY_EXISTS == GetLastError())
    {
        // CopyFile(...)
        std::cout << "Creating results directory..." << std::endl;
    }
    else
    {
        // Failed to create directory.
        std::cout << "Could not create directory" << std::endl;
    }

    for (int iter = 0; iter < nImages; iter++) {
        std::cout << "image " << iter << std::endl;

        // Image to read
        std::string image_path = path + folder + std::to_string(iter) + ".tif";
        img = cv::imread(image_path);

        // Make sure the image is read
        if (img.empty())
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }
        // Resize the image (NOTE: cv::cuda::resize has an iterpolation bug which is why I'm doing this outside of the if statement)
        if (SCALE != 1.0f) {
            cv::resize(img, img, cv::Size(img.cols * SCALE, img.rows * SCALE));// , SCALE, SCALE);
        }

        // Store the color scaled image
        img.copyTo(imgColor);
        cv::Mat arucoFrame;
        img.copyTo(arucoFrame);

        // Convert to grayscale
        if (img.channels() == 3) {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        // Storage for corners, ids found by ardvrc method
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;

        // Detect markers using ardvrc (and time it!)
        begin = std::chrono::steady_clock::now();
        ardvrc.detectMarkers(img, corners, ids);
        end = std::chrono::steady_clock::now();

        // Store ArDVRC timing
        ardvrcTime[iter] = ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0f;
        if (ids.size() > 0) {

            // Store frame number
            ardvrcFrames.push_back(iter);

            // Store the ID of the marker
            ardvrcIds.push_back(ids[0]);

            // Store the detected corners of the marker
            for (int i = 0; i < 4; i++) {
                ardvrcCorners.push_back(corners[0][i].x);
                ardvrcCorners.push_back(corners[0][i].y);
            }
        }

        // Drawing code (corner vectors, detected markers)
        //ardvrc.drawCornerVectors(imgColor, 10, true, true);
        //cv::aruco::drawDetectedMarkers(imgColor, corners, ids, cv::Scalar(255, 255, 0));

        std::vector<std::vector<cv::Point2f>> arucocorners;
        std::vector<int> arucoids;
        
        // Detect markers using ArUco method (and time it!)
        begin = std::chrono::steady_clock::now();
        cv::aruco::detectMarkers(img, ardvrc.dictionary, arucocorners, arucoids, ardvrc.parameters);
        end = std::chrono::steady_clock::now();

        // Store aruco timing
        arucoTime[iter] = ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0f;
        if (arucoids.size() > 0) {

            // Save frame number
            arucoFrames.push_back(iter);

            // Save marker ID
            arucoIds.push_back(arucoids[0]);

            // Save all of the corners
            for (int i = 0; i < 4; i++) {
                arucoCorners.push_back(arucocorners[0][i].x);
                arucoCorners.push_back(arucocorners[0][i].y);
            }

        }

        /*cv::aruco::drawDetectedMarkers(arucoFrame, arucocorners, arucoids);
        cv::imshow("Display window", imgColor);
        cv::imshow("ArUco", arucoFrame);

        int k = cv::waitKey(0); // Wait for a keystroke in the window
        if (k == 'q') return 0;
        if (k == 's') cv::imwrite("output.png", img);*/
    }

    // Save ID information
    cnpy::npy_save(outfolder + "aruco_ids.npy", arucoIds);
    cnpy::npy_save(outfolder + "ardvrc_ids.npy", ardvrcIds);

    // Save frame information
    cnpy::npy_save(outfolder + "aruco_frames.npy", arucoFrames);
    cnpy::npy_save(outfolder + "ardvrc_frames.npy", ardvrcFrames);

    // Store corner information
    cnpy::npy_save(outfolder + "aruco_corners.npy", arucoCorners);
    cnpy::npy_save(outfolder + "ardvrc_corners.npy", ardvrcCorners);

    // Save timing information
    cnpy::npy_save(outfolder + "aruco_time.npy", arucoTime);
    cnpy::npy_save(outfolder + "ardvrc_time.npy", ardvrcTime);

    return 0;
}