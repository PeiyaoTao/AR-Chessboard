cmake_minimum_required(VERSION 3.8)
project( Project4 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

set(GLFW_INCLUDE_DIR "/opt/homebrew/opt/glfw/include")
set(GLFW_LIBRARY "/opt/homebrew/opt/glfw/lib/libglfw.dylib")
include_directories(${GLFW_INCLUDE_DIR})

set(GLUT_LIBRARY "-framework GLUT")
set(OPENGL_LIBRARY "-framework OpenGL")

set(GLEW_INCLUDE_DIR "/opt/homebrew/opt/glew/include")
set(GLEW_LIBRARY "/opt/homebrew/opt/glew/lib/libGLEW.dylib")
include_directories(${GLEW_INCLUDE_DIR})

add_definitions(-DGL_SILENCE_DEPRECATION)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CALIBRATION
    src/calibration.cpp
)

add_executable( Calibration ${CALIBRATION} )
target_link_libraries( Calibration ${OpenCV_LIBS} )

set(CHECKERBOARD_AR
    src/checkerboardAR.cpp
)

add_executable( CheckerboardAR ${CHECKERBOARD_AR} )
target_link_libraries( CheckerboardAR ${OpenCV_LIBS} )

set(EXTENSION_AR
    src/extensionAR.cpp
)

add_executable( ExtensionAR ${EXTENSION_AR} )
target_link_libraries(ExtensionAR ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLFW_LIBRARY} ${GLEW_LIBRARY} ${GLUT_LIBRARY})

set(HARRIS_CORNERS
    src/harrisCorners.cpp
)

add_executable( HarrisCorners ${HARRIS_CORNERS} )
target_link_libraries( HarrisCorners ${OpenCV_LIBS} )