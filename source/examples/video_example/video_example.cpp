// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <chrono>
#include <numeric>
#include <limits>
#include "ArDVRC.h"

int main(int argc, char* argv[])
{

    std::cout << "\n*** Welcome to the ArDVRC Video Example. ***\n\n";
    std::cout << "Usage:\n\n";
    std::cout << "\tRunning sample video:\n\n\t\t./video_example.exe\n\n";
    std::cout << "\tRunning with a user-provided video:\n\n\t\t./video_example.exe --file <filename>\n\n";
    std::cout << "\tRunning with live video:\n\n\t\t./video_example.exe --live\n\n";
    std::cout << "\tRunning with side-by-side ArUco comparison:\n\n\t\t./video_example.exe --compare\n\n";
    std::cout << "\tShow ArDVRC directed corner vectors:\n\n\t\t./video_example.exe --vectors\n\n";
    // TODO: Writing to a file needs more work before it's ready as a flag, but it can be uncommented in the program
    //std::cout << "\tWrite video to file:\n\n\t\t./video_example.exe --write\n\n";
    std::cout << "\tPress 'ESC' to exit.\n\n";
    std::cout << "\tNOTE: If you choose both --live and --file, live mode will run\n";
    std::cout << "\tinstead of the file.\n\n";

    bool live_video = false;
    bool write_video = false;
    bool compare = false;
    bool vectors = false;
    std::string filename = "example_video.mp4";

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            // TODO: Needs to account for different framerates and video sizes before it can be a flag
            /*if (std::string(argv[i]) == "--write") {
                std::cout << "Video will be written to output.avi.\n";
                write_video = true;
            }*/
            if (std::string(argv[i]) == "--live") {
                live_video = true;
                std::cout << "Using live video stream from first camera available.\n";
            }
            if (std::string(argv[i]) == "--compare") {
                compare = true;
                std::cout << "Comparing with ArUco processing.\n";
            }
            if (std::string(argv[i]) == "--vectors") {
                vectors = true;
                std::cout << "Comparing with ArUco processing.\n";
            }
            if (std::string(argv[i]) == "--file")
            {
                if (i + 1 == argc) {
                    std::cout << "Please provide a filename if you are going to use the --file option.\n";
                }
                else {
                    std::cout << "Using " << argv[i + 1] << " as video file.\n";
                    filename = std::string(argv[i + 1]);
                }
            }
        }
    }
    

    // Image scaling factor
    const float SCALE = 0.5f;//0.5f normally
    const bool BLUR = true;
    const int BLUR_KERNEL = 3;
    const int RADIUS = 4;
    const int N = 5;

    ArDVRC ardvrc(N, BLUR, BLUR_KERNEL, RADIUS, RADIUS, "DICT_4X4_64_ENTROPY.npy",0.00015f);
    

    cv::Mat img, imgColor;

    cv::VideoCapture cap;

    // Choose the file
    if (!live_video) {
        cap = cv::VideoCapture(filename);
    }
    // Choose the live stream. If multiple cameras plugged in, you may want to change 0 to a different number
    else {
        cap = cv::VideoCapture(0);
    }

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // You can uncomment this to play with video writing. Make sure to also uncomment calls to out.write() and out.release()
    // Make sure you adjust the frames per second, image size, etc. to match your video stream
    //cv::VideoWriter out = cv::VideoWriter("outputvideo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 60.0, cv::Size(1280, 720 / 2));

    // ArUco processing (if --compare)
    cv::Mat arucoFrame;

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

        if (compare) {
            frame.copyTo(arucoFrame);
        }

        // ArDVRC processing
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        ardvrc.detectMarkers(gray, corners, ids);
        cv::aruco::drawDetectedMarkers(frame, corners, ids, cv::Scalar(255, 255, 0));

        // Uncomment this to view direction vectors
        if (vectors) {
            ardvrc.drawCornerVectors(frame, 10, true, true, true);
        }

        if (compare) {
            
            std::vector<std::vector<cv::Point2f>> arucocorners;
            std::vector<int> arucoids;
            cv::aruco::detectMarkers(gray, ardvrc.dictionary, arucocorners, arucoids, ardvrc.parameters);
            cv::aruco::drawDetectedMarkers(arucoFrame, arucocorners, arucoids);

            cv::Mat output;
            cv::putText(arucoFrame, "ArUco", cv::Point(10, 20), 0, 0.5, cv::Scalar(255, 0, 0));
            cv::putText(frame, "ArDVRC", cv::Point(10, 20), 0, 0.5, cv::Scalar(255, 0, 0));
            cv::hconcat(arucoFrame, frame, output);

            // Display the resulting frame
            cv::imshow("Output", output);
        }
        else {
            cv::imshow("Output", frame);
        }

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
