/*
  Peiyao Tao
  CS 5330
  Project 4 Extension AR
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

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

    cv::FileStorage file("intrinsic_parameters.yml", cv::FileStorage::READ);
    file["camera_matrix"] >> camera_matrix;
    file["distortion_coefficients"] >> distortion_coefficients;

    cv::VideoCapture cap(1);

    if (!cap.isOpened())
    {
        printf("Could not open camera.\n");
        return -1;
    }

    float squareSize = 1.0f;
    // Set minionSize to 70% of a square.
    float minionSize = 0.7f * squareSize;

    std::vector<cv::Point2f> corner_set;

    std::vector<cv::Point3f> point_set;
    for (int i = 0; i < INTERNAL_CORNER_ROWS; i++)
    {
        for (int j = 0; j < INTERNAL_CORNER_COLUMNS; j++)
        {
            point_set.push_back(cv::Point3f(j * squareSize + 1, i * squareSize + 1, 0));
        }
    }

    // Minion object that contains the position and side of minion.
    struct Minion
    {
        cv::Point3f position;
        // two types of minions on each side.
        int type;
        cv::Scalar color;
    };

    std::vector<Minion> minions;

    // Left Side Minions
    for (int i = 0; i < INTERNAL_CORNER_ROWS + 1; i++)
    {
        minions.push_back({cv::Point3f(squareSize / 2, i * squareSize + squareSize / 2, 0), 0, cv::Scalar(255, 0, 0)});
        minions.push_back({cv::Point3f(squareSize, i * squareSize, 0), 1, cv::Scalar(255, 0, 0)});
    }

    // Right Side Minions
    for (int i = 0; i < INTERNAL_CORNER_ROWS + 1; i++)
    {
        minions.push_back({cv::Point3f(INTERNAL_CORNER_COLUMNS * squareSize + (squareSize / 2), i * squareSize + (squareSize / 2), 0), 0, cv::Scalar(0, 0, 255)});
        minions.push_back({cv::Point3f((INTERNAL_CORNER_COLUMNS - 1) * squareSize, i * squareSize, 0), 1, cv::Scalar(0, 0, 255)});
    }

    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            printf("The frame is empty.\n");
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Fast check for chessboard
        bool patternWasFound = cv::findChessboardCorners(
            gray,
            cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS),
            corner_set,
            cv::CALIB_CB_FAST_CHECK);

        if (patternWasFound)
        {
            // Run more complex detection if fast check found.
            // Do this to speed up the program when no pattern found.
            patternWasFound = cv::findChessboardCorners(
                gray,
                cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS),
                corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

            if (patternWasFound)
            {
                cv::cornerSubPix(gray, corner_set, cv::Size(5, 5), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001));

                cv::Mat rotation, translation;
                cv::solvePnP(point_set, corner_set, camera_matrix, distortion_coefficients, rotation, translation);

                // Project and draw minions
                for (const auto &minion : minions)
                {
                    std::vector<cv::Point3f> objectPoints;
                    std::vector<cv::Point2f> imagePoints;

                    if (minion.type == 0)
                    {
                        // Minion that consists by a cylinder with a sphere on top.

                        // Draw the cylinder
                        // Use 16 points to draw a raw circle.
                        int numCirclePoints = 16;
                        float cylinderRadius = minionSize / 2.0f;
                        float cylinderHeight = minionSize;
                        std::vector<cv::Point3f> baseCircle;
                        std::vector<cv::Point3f> topCircle;
                        for (int i = 0; i < numCirclePoints; ++i)
                        {
                            float angle = 2 * CV_PI * i / numCirclePoints;
                            float x = cylinderRadius * cos(angle);
                            float y = cylinderRadius * sin(angle);
                            // Draw the base and top circles of a cylinder.
                            baseCircle.push_back(cv::Point3f(minion.position.x + x, minion.position.y + y, minion.position.z));
                            topCircle.push_back(cv::Point3f(minion.position.x + x, minion.position.y + y, minion.position.z - cylinderHeight));
                        }

                        // Draw the sphere
                        int numSphereLayers = 5;
                        float sphereRadius = cylinderRadius;
                        float sphereCenterZAxis = minion.position.z - cylinderHeight - sphereRadius;
                        std::vector<std::vector<cv::Point3f>> sphereLayers;
                        for (int i = 0; i < numSphereLayers; ++i)
                        {
                            float phi = CV_PI * i / (numSphereLayers - 1);
                            float layerRadius = sphereRadius * sin(phi);
                            float z = sphereCenterZAxis + sphereRadius * cos(phi);
                            std::vector<cv::Point3f> layer;
                            for (int j = 0; j < numCirclePoints; ++j)
                            {
                                float theta = 2 * CV_PI * j / numCirclePoints;
                                float x = layerRadius * cos(theta);
                                float y = layerRadius * sin(theta);
                                layer.push_back(cv::Point3f(minion.position.x + x, minion.position.y + y, z));
                            }
                            sphereLayers.push_back(layer);
                        }
                        // Combine all points
                        objectPoints.insert(objectPoints.end(), baseCircle.begin(), baseCircle.end());
                        objectPoints.insert(objectPoints.end(), topCircle.begin(), topCircle.end());
                        for (const auto &layer : sphereLayers)
                        {
                            objectPoints.insert(objectPoints.end(), layer.begin(), layer.end());
                        }
                        cv::projectPoints(objectPoints, rotation, translation, camera_matrix, distortion_coefficients, imagePoints);
                        // Draw the cylinder edges
                        int offset = 0;
                        for (int i = 0; i < numCirclePoints; ++i)
                        {
                            int next_i = (i + 1) % numCirclePoints;
                            cv::line(frame, imagePoints[offset + i], imagePoints[offset + next_i], minion.color, 1);
                            cv::line(frame, imagePoints[offset + numCirclePoints + i], imagePoints[offset + numCirclePoints + next_i], minion.color, 1);
                            cv::line(frame, imagePoints[offset + i], imagePoints[offset + numCirclePoints + i], minion.color, 1);
                        }
                        offset += 2 * numCirclePoints;
                        // Draw sphere layers
                        for (int l = 0; l < numSphereLayers; ++l)
                        {
                            for (int i = 0; i < numCirclePoints; ++i)
                            {
                                int next_i = (i + 1) % numCirclePoints;
                                int idx = offset + l * numCirclePoints + i;
                                int next_idx = offset + l * numCirclePoints + next_i;
                                cv::line(frame, imagePoints[idx], imagePoints[next_idx], minion.color, 1);
                                // Connect layers
                                if (l < numSphereLayers - 1)
                                {
                                    int idx_next_layer = offset + (l + 1) * numCirclePoints + i;
                                    cv::line(frame, imagePoints[idx], imagePoints[idx_next_layer], minion.color, 1);
                                }
                            }
                        }
                    }
                    else
                    {
                        // Minion that consists by a cuboid with a pyramid on top.
                        float cuboidWidth = minionSize;
                        float cuboidLength = minionSize;
                        float cuboidHeight = minionSize;
                        cv::Point3f cuboidPos = minion.position;
                        std::vector<cv::Point3f> cuboidPoints = {
                            cuboidPos,
                            cuboidPos + cv::Point3f(cuboidWidth, 0, 0),
                            cuboidPos + cv::Point3f(cuboidWidth, cuboidLength, 0),
                            cuboidPos + cv::Point3f(0, cuboidLength, 0),
                            cuboidPos + cv::Point3f(0, 0, -cuboidHeight),
                            cuboidPos + cv::Point3f(cuboidWidth, 0, -cuboidHeight),
                            cuboidPos + cv::Point3f(cuboidWidth, cuboidLength, -cuboidHeight),
                            cuboidPos + cv::Point3f(0, cuboidLength, -cuboidHeight),
                        };

                        // The apex of pyramid.
                        cv::Point3f apex = cuboidPos + cv::Point3f(cuboidWidth / 2, cuboidLength / 2, -cuboidHeight - minionSize);

                        objectPoints.insert(objectPoints.end(), cuboidPoints.begin(), cuboidPoints.end());
                        objectPoints.push_back(apex);
                        cv::projectPoints(objectPoints, rotation, translation, camera_matrix, distortion_coefficients, imagePoints);

                        // Draw cuboid.
                        int idx[8];
                        for (int i = 0; i < 8; ++i)
                        {
                            idx[i] = i;
                        }

                        cv::line(frame, imagePoints[idx[0]], imagePoints[idx[1]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[1]], imagePoints[idx[2]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[2]], imagePoints[idx[3]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[3]], imagePoints[idx[0]], minion.color, 1);

                        cv::line(frame, imagePoints[idx[4]], imagePoints[idx[5]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[5]], imagePoints[idx[6]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[6]], imagePoints[idx[7]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[7]], imagePoints[idx[4]], minion.color, 1);

                        cv::line(frame, imagePoints[idx[0]], imagePoints[idx[4]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[1]], imagePoints[idx[5]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[2]], imagePoints[idx[6]], minion.color, 1);
                        cv::line(frame, imagePoints[idx[3]], imagePoints[idx[7]], minion.color, 1);

                        // Draw pyramid.
                        int apexIdx = 8;
                        for (int i = 4; i < 8; ++i)
                        {
                            cv::line(frame, imagePoints[i], imagePoints[apexIdx], minion.color, 1);
                        }
                    }
                }

                // cv::drawChessboardCorners(frame, cv::Size(INTERNAL_CORNER_COLUMNS, INTERNAL_CORNER_ROWS), cv::Mat(corner_set), patternWasFound);
            }
            else
            {
                printf("Chessboard not found in full detection.\n");
            }
        }
        else
        {
            printf("Chessboard not found in fast check.\n");
        }

        cv::imshow("Target Detection", frame);

        char key = cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
