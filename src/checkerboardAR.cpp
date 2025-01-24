/*
  Peiyao Tao
  CS 5330
  Project 4 Checkerboard AR (Task 4-6)
*/

#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <iostream>

/* Number of inner corners per row (num colunms). */
const int INTERNAL_CORNER_COLUMNS = 9;
/* Number of inner corners per column (num rows). */
const int INTERNAL_CORNER_ROWS = 6;

/**
 * Main function that use the intrinsic parameters and apply AR on top of the chessboard.
 */
int main(int argc, char **argv)
{
    cv::Mat camera_matrix, frame, gray;
    std::vector<double> distortion_coefficients;

    // Use openCV to read .yml file.
    cv::FileStorage file("intrinsic_parameters.yml", cv::FileStorage::READ);
    file["camera_matrix"] >> camera_matrix;
    file["distortion_coefficients"] >> distortion_coefficients;

    cv::VideoCapture cap(1);

    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return -1;
    }

    std::vector<cv::Point2f> corner_set;

    // Initialize point_set, storing the corner locations.
    std::vector<cv::Vec3f> point_set;
    for (int i = 0; i < INTERNAL_CORNER_ROWS; i++)
    {
        for (int j = 0; j < INTERNAL_CORNER_COLUMNS; j++)
        {
            point_set.push_back(cv::Vec3f(j + 1, -(i + 1), 0));
        }
    }

    // Set up the points for drawing axes on the chessboard.
    int axisLength = 5;
    std::vector<cv::Point3f> axesPoints;
    axesPoints.push_back(cv::Point3f(0, 0, 0));
    axesPoints.push_back(cv::Point3f(axisLength, 0, 0));
    axesPoints.push_back(cv::Point3f(0, -axisLength, 0));
    axesPoints.push_back(cv::Point3f(0, 0, -axisLength));

    // Points of the four outer corners of chessboard.
    std::vector<cv::Point3f> checkerboardCornerPoints;
    checkerboardCornerPoints.push_back(cv::Point3f(0, 0, 0));
    checkerboardCornerPoints.push_back(cv::Point3f(INTERNAL_CORNER_COLUMNS + 1, 0, 0));
    checkerboardCornerPoints.push_back(cv::Point3f(INTERNAL_CORNER_COLUMNS + 1, -(INTERNAL_CORNER_ROWS + 1), 0));
    checkerboardCornerPoints.push_back(cv::Point3f(0, -(INTERNAL_CORNER_ROWS + 1), 0));

    // Create a pyramid above the checkerboard.
    std::vector<cv::Point3f> pyramidPoints;
    pyramidPoints.push_back(cv::Point3f(2, -2, 2));
    pyramidPoints.push_back(cv::Point3f(6, -2, 2));
    pyramidPoints.push_back(cv::Point3f(6, -6, 2));
    pyramidPoints.push_back(cv::Point3f(2, -6, 2));
    pyramidPoints.push_back(cv::Point3f(4, -4, 5));

    std::vector<std::pair<int, int>> pyramidEdges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};

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

        // boolean that says if chessboard pattern was found.
        bool patternWasFound = cv::findChessboardCorners(gray, cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS), corner_set, cv::CALIB_CB_FAST_CHECK);

        if (patternWasFound)
        {
            cv::cornerSubPix(gray, corner_set, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001));

            cv::Mat rotation, translation;
            cv::solvePnP(point_set, corner_set, camera_matrix, distortion_coefficients, rotation, translation);

            // Project axes on checkerboard.
            std::vector<cv::Point2f> axesOutputPoints;
            cv::projectPoints(axesPoints, rotation, translation, camera_matrix, distortion_coefficients, axesOutputPoints);

            // Draw the axes.
            cv::line(frame, axesOutputPoints[0], axesOutputPoints[1], cv::Scalar(0, 255, 0), 3);
            cv::line(frame, axesOutputPoints[0], axesOutputPoints[2], cv::Scalar(0, 0, 255), 3);
            cv::line(frame, axesOutputPoints[0], axesOutputPoints[3], cv::Scalar(255, 0, 0), 3);

            // Project checkerboard corners on checkerboard.
            std::vector<cv::Point2f> cornerOutputPoints;
            cv::projectPoints(checkerboardCornerPoints, rotation, translation, camera_matrix, distortion_coefficients, cornerOutputPoints);

            // Draw circles on the corners.
            for (int i = 0; i < cornerOutputPoints.size(); i++)
            {
                cv::circle(frame, cornerOutputPoints[i], 5, cv::Scalar(100, 100, 100));
            }

            // Connect checkerboard corners.
            cv::line(frame, cornerOutputPoints[0], cornerOutputPoints[1], cv::Scalar(200, 200, 0), 2);
            cv::line(frame, cornerOutputPoints[1], cornerOutputPoints[2], cv::Scalar(200, 200, 0), 2);
            cv::line(frame, cornerOutputPoints[2], cornerOutputPoints[3], cv::Scalar(200, 200, 0), 2);
            cv::line(frame, cornerOutputPoints[3], cornerOutputPoints[0], cv::Scalar(200, 200, 0), 2);

            // Project pyramid on the checkerboard.
            std::vector<cv::Point2f> pyramidImagePoints;
            cv::projectPoints(pyramidPoints, rotation, translation, camera_matrix, distortion_coefficients, pyramidImagePoints);

            // Connect pyramid points.
            for (auto &edge : pyramidEdges)
            {
                cv::line(frame, pyramidImagePoints[edge.first], pyramidImagePoints[edge.second], cv::Scalar(255, 0, 255), 3);
            }

            std::cout << "Rotation Matrix:\n"
                      << rotation << std::endl;
            std::cout << "Translation Vector:\n"
                      << translation << std::endl;

            // cv::drawChessboardCorners(frame, cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS), cv::Mat(corner_set), patternWasFound);
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
        else if (key == 's')
        {
            std::string imgName = "checkerboard_AR_" + std::to_string(count) + ".jpg";
            std::string outputDirectory = "../";
            cv::imwrite(outputDirectory + imgName, frame);
            count++;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}