/*
  Peiyao Tao
  CS 5330
  Project 4 Calibration (Task 1-3)
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <iomanip>

/* Number of inner corners per row (num colunms). */
const int INTERNAL_CORNER_COLUMNS = 9;
/* Number of inner corners per column (num rows). */
const int INTERNAL_CORNER_ROWS = 6;
/* Minimum numbe of saved calibration image needed. */
const int MIN_SAVED_CALIBRATION = 5;

/**
 * A helper function that prints the distortion coefficients to output stream.
 */
inline void printDistortionCoefficients(std::vector<double> &distortion_coefficients, std::ostream &out)
{
    for (double value : distortion_coefficients)
    {
        out << value << " ";
    }
}

/**
 * Main function that displays the video and apply calibration to each frame.
 */
int main(int argc, char **argv)
{
    cv::VideoCapture cap(1);
    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return -1;
    }

    cv::Mat frame, gray, lastCornerFoundFrame;
    std::vector<cv::Point2f> corner_set, lastFoundCornerSet;

    // Initialize point_set, storing the corner locations.
    std::vector<cv::Vec3f> point_set;
    for (int i = 0; i < INTERNAL_CORNER_ROWS; i++)
    {
        for (int j = 0; j < INTERNAL_CORNER_COLUMNS; j++)
        {
            point_set.push_back(cv::Vec3f(j, -i, 0));
        }
    }

    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;

    cv::Mat camera_matrix;
    bool cameraMatrixInitialized = false;
    std::vector<double> distortion_coefficients;
    std::vector<cv::Mat> rotations, translations;

    int count = 0;
    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            printf("The frame is empty.\n");
            break;
        }

        // If camera matrix is not initialized (no calibration before), initialize camera matrix.
        // Initialize here because we need the frame.cols and frame.rows
        if (!cameraMatrixInitialized)
        {
            camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, frame.cols / 2,
                             0, 1, frame.rows / 2,
                             0, 0, 1);
            cameraMatrixInitialized = true;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // boolean that says if chessboard pattern was found.
        bool patternWasFound = cv::findChessboardCorners(gray, cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS), corner_set,
                                                         cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if (patternWasFound)
        {
            cv::cornerSubPix(gray, corner_set, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001));

            // draw the corners on frame.
            cv::drawChessboardCorners(frame, cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS), cv::Mat(corner_set), patternWasFound);

            // store this frame and correspond corner set. If the user press 's' to save and the program does not find a pattern on that frame,
            // it will use these stored last frame and last corner set to store.
            lastCornerFoundFrame.create(frame.size(), frame.type());
            frame.copyTo(lastCornerFoundFrame);
            lastFoundCornerSet = corner_set;

            printf("%zu corners found.\n", corner_set.size());
            printf("Coordinate of first corner: %f, %f\n", corner_set[0].x, corner_set[0].y);
        }
        else
        {
            printf("Checkrboard not found.\n");
        }

        cv::imshow("Target Detection", frame);

        char key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
        // If the user wants to save.
        else if (key == 's')
        {
            // If there is something to store.
            if (patternWasFound || !lastFoundCornerSet.empty())
            {
                std::string imgName = "seleted_calibration_image_" + std::to_string(count) + ".jpg";
                std::string outputDirectory = "../";

                // If pattern not found in current frame, save stored previous frame.
                if (!patternWasFound)
                {
                    corner_list.push_back(lastFoundCornerSet);
                    cv::imwrite(outputDirectory + imgName, lastCornerFoundFrame);
                }
                else
                {
                    corner_list.push_back(corner_set);
                    cv::imwrite(outputDirectory + imgName, frame);
                }
                count++;
                point_list.push_back(point_set);
            }
            else
            {
                printf("No corners detected.\n");
            }
        }
        // If the user wants to calibrate.
        else if (key == 'c')
        {
            // If the user saved more than required number of calibration images.
            if (corner_list.size() >= MIN_SAVED_CALIBRATION)
            {
                // Print before calibration.
                std::cout << "Camera Matrix Before Calibration:\n"
                          << camera_matrix
                          << std::endl;
                std::cout << "Distortion Coefficients Before Calibration:\n";
                printDistortionCoefficients(distortion_coefficients, std::cout);
                std::cout << std::endl;

                // Calibrate the camera and get the re-projection error.
                double error = cv::calibrateCamera(point_list, corner_list, frame.size(),
                                                   camera_matrix, distortion_coefficients,
                                                   rotations, translations,
                                                   cv::CALIB_FIX_ASPECT_RATIO);
                // Print after calibration.
                std::cout << "Camera Matrix After Calibration:\n"
                          << camera_matrix
                          << std::endl;
                std::cout << "Distortion Coefficients After Calibration:\n";
                printDistortionCoefficients(distortion_coefficients, std::cout);
                std::cout << std::endl;
                std::cout << "Final re-projection error:\n"
                          << error
                          << std::endl;
            }
            else
            {
                printf("Require at least 5 calibration frames.\n");
                printf("Press 's' to save calibration frames.\n");
                printf("%zu calibration frames saved.\n", corner_list.size());
            }
        }
        // If the user wants to write the intrinsic parameters to a file.
        else if (key == 'w')
        {
            // save to .yml format to be easily read by openCV.
            cv::FileStorage file("intrinsic_parameters.yml", cv::FileStorage::WRITE);
            if (file.isOpened())
            {
                file << "camera_matrix" << camera_matrix;
                file << "distortion_coefficients" << distortion_coefficients;
            }
            else
            {
                printf("Failed to write instrinsic_parameters.");
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
