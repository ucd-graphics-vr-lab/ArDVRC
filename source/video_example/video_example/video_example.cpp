// custom_aruco.cpp : This file contains the 'main' function. Program execution begins and ends there.

// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <numeric>
#include <limits>
#include "../../../include/ArDVRC.h"

int main()
{
    // Image scaling factor
    const float SCALE = 0.5f;//0.5f normally
    const bool BLUR = true;
    const int BLUR_KERNEL = 3;
    const int RADIUS = 4;
    const int N = 5;

    ArDVRC ardvrc(N, BLUR, BLUR_KERNEL, RADIUS, RADIUS, "..\\..\\..\\data\\entropy_dictionary\\DICT_4X4_64_ENTROPY.npy",0.00015f);

    // Six different colors used for drawing
    std::vector<cv::Scalar> colors(6);
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    colors[3] = cv::Scalar(255, 255, 0);
    colors[4] = cv::Scalar(255, 0, 255);
    colors[5] = cv::Scalar(0, 255, 255);

    cv::Mat img, imgColor;

    // Sample recorded video
    cv::VideoCapture cap("..\\..\\..\\data\\sample_data\\example_video.mp4");

    // Use this for real-time webcam video
    //cv::VideoCapture cap(0);

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Uncomment references to out if you'd like to write a video
    // Make sure you adjust the frames per second, image size, etc. to match your video stream
    //cv::VideoWriter out = cv::VideoWriter("outputvideo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60.0, cv::Size(1280, 720 / 2));

    while (1) {

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;


        // Resize the image (NOTE: cv::cuda::resize has an iterpolation bug which is why I'm doing this outside of the if statement)
        if (SCALE != 1.0f) {
            cv::resize(frame, frame, cv::Size(frame.cols * SCALE, frame.rows * SCALE));// , SCALE, SCALE);
        }

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat arucoFrame;
        frame.copyTo(arucoFrame);

        //begin = std::chrono::steady_clock::now();
        std::vector<std::vector<cv::Point2f>> arucocorners;
        std::vector<int> arucoids;
        cv::aruco::detectMarkers(gray, ardvrc.dictionary, arucocorners, arucoids, ardvrc.parameters);
        cv::aruco::drawDetectedMarkers(arucoFrame, arucocorners, arucoids);

        // Uncomment this to view direction vectors
        //ardvrc.drawCornerVectors(frame, 10, true, true, true);

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        ardvrc.detectMarkers(gray, corners, ids);
        cv::aruco::drawDetectedMarkers(frame, corners, ids, cv::Scalar(255, 255, 0));

        cv::Mat output;
        cv::putText(arucoFrame, "ArUco", cv::Point(10, 20), 0, 0.5, cv::Scalar(255, 0, 0));
        cv::putText(frame, "ArDVRC", cv::Point(10, 20), 0, 0.5, cv::Scalar(255, 0, 0));
        cv::hconcat(arucoFrame, frame, output);

        // Display the resulting frame
        cv::imshow("Output", output);

        //out.write(output);

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(5);
        if (c == 27)
            break;
    }

    //out.release();
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}