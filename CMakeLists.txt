cmake_minimum_required(VERSION 3.10)

project(COMPUTERVISION-FINALPROJECT)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(main 
    src/main.cpp
    src/DatasetLoader.cpp
    src/FeatureExtractor.cpp
    src/Classifier.cpp
    src/tracker.cpp
    src/YoloTracker.cpp
)

# Include path
target_include_directories(main
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# Link OpenCV
target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
)