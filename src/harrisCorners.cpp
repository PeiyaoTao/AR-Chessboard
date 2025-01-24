/*
  Peiyao Tao
  CS 5330
  Project 4 Harris Corners (Task 7)
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>

/**
 * Main function that use Harris corner to detect corners of blocks.
 */
int main(int argc, char **argv)
{
    cv::Mat frame, gray;

    cv::VideoCapture cap(1);

    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return -1;
    }

    int count = 0;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            printf("The frame is empty.\n");
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Apply a blur to reduce noise.
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        cv::Mat dst = cv::Mat::zeros(gray.size(), CV_32FC1);
        cv::cornerHarris(gray, dst, 3, 3, 0.04);

        cv::Mat dst_norm, dst_norm_scaled;
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        int threshold = 200;
        for (int i = 0; i < dst_norm.rows; i++)
        {
            for (int j = 0; j < dst_norm.cols; j++)
            {
                if ((int)dst_norm.at<float>(i, j) > threshold)
                {
                    cv::circle(frame, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        cv::imshow("Target Detection", frame);

        char key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            std::string imgName = "harris_corners_" + std::to_string(count) + ".jpg";
            std::string outputDirectory = "../";
            cv::imwrite(outputDirectory + imgName, frame);
            count++;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}