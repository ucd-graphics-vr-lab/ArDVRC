// Make sure M_PI is defined so we can use it
#define _USE_MATH_DEFINES

// Whether to use GPU for image processing or not
#define USE_GPU 0

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

    std::cout << "\n*** Welcome to the ArDVRC Image Example. ***\n\n";
    std::cout << "Usage:\n\n";
    std::cout << "\tRunning sample images:\n\n\t\t./image_example.exe\n\n";
    std::cout << "\tRunning custom images:\n\n\t\t./image_example.exe <image0.jpg> <image1.jpg> ... <imagen.jpg>\n\n";
    std::cout << "\tPress 'ESC' to exit.\n\n";

    std::vector<std::string> image_paths;

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            image_paths.push_back(std::string(argv[i]));
     
        }
    }
    else {
        image_paths.push_back("example_image1.jpg");
        image_paths.push_back("example_image2.jpg");
        image_paths.push_back("example_image3.jpg");
        image_paths.push_back("example_image4.jpg");
    }

    // Image scaling factor
    const float SCALE = 0.5f;
    const bool BLUR = true;
    const int BLUR_KERNEL = 3;
    const int RADIUS = 4;
    const int ROI_SHAPE = RADIUS * 2 + 1;
    const int N = 5;

    ArDVRC ardvrc(N, BLUR, BLUR_KERNEL, RADIUS, RADIUS, "DICT_4X4_64_ENTROPY.npy", 0.00015f);
    cv::Mat img, imgColor;

    cv::Mat outImg, outAruco;

    
    for (int i = 0; i < image_paths.size(); i++) {

        img = cv::imread(image_paths[i]);
        img.copyTo(imgColor);

        // Make sure the image is read
        if (img.empty())
        {
            std::cout << "Could not read the image: " << image_paths[i] << std::endl;
            return 1;
        }
        // Resize the image (example images are very large, need to be resize to 1/10 of the width/height)
        cv::resize(img, img, cv::Size(img.cols * SCALE, img.rows * SCALE));

        // Store the color scaled image
        img.copyTo(imgColor);
        cv::Mat arucoFrame;
        img.copyTo(arucoFrame);

        // Convert to grayscale
        if (img.channels() == 3) {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        // run ardvrc detection
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        ardvrc.detectMarkers(img, corners, ids);

        //ardvrc.drawCornerVectors(imgColor, 10, true, true);
        cv::aruco::drawDetectedMarkers(imgColor, corners, ids, cv::Scalar(255, 255, 0));

        std::vector<std::vector<cv::Point2f>> arucocorners;
        std::vector<int> arucoids;
        cv::aruco::detectMarkers(img, ardvrc.dictionary, arucocorners, arucoids, ardvrc.parameters);

        cv::aruco::drawDetectedMarkers(arucoFrame, arucocorners, arucoids);

        int k = cv::waitKey(0); // Wait for a keystroke in the window

        if (i == 0) {
            outImg = imgColor;
            outAruco = arucoFrame;
        }
        else {
            cv::hconcat(outImg, imgColor, outImg);
            cv::hconcat(outAruco, arucoFrame, outAruco);
        }

    }
    cv::imshow("ArDVRC Results", outImg);
    cv::imshow("ArUco Results", outAruco);

    int k = cv::waitKey(0); // Wait for a keystroke in the window
    
    // Uncomment to write output image                        
    cv::imwrite("outputArDVRC.png", outImg);
    cv::imwrite("outputArUco.png", outAruco);

    return 0;
}