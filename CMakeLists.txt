cmake_minimum_required(VERSION 3.5)
project(fast_lio)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Eigen REQUIRED eigen3)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
# find_package(eigen_conversions REQUIRED)
# find_package(message_generation REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(OpenMP QUIET)
find_package(capella_ros_msg REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

include_directories(
  include
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

#添加消息文件和依赖
rosidl_generate_interfaces(${PROJECT_NAME}
  #---msg---
  "msg/Pose6D.msg"

  #---srv---
  
  DEPENDENCIES geometry_msgs
 )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

#laserMapping_node
add_executable(laserMapping_node src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
ament_target_dependencies(laserMapping_node 
                          rclcpp
                          tf2
                          tf2_ros
                          sensor_msgs
                          std_msgs
                          geometry_msgs
                          pcl_ros
                          livox_ros_driver2
                          nav_msgs
                          tf2_geometry_msgs
                          capella_ros_msg)

target_link_libraries(laserMapping_node
                      ${colcon_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${PYTHON_LIBRARIES}
)

target_include_directories(laserMapping_node
   PRIVATE 
    ${Eigen_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    src/
)

install(TARGETS
        laserMapping_node
        DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz_cfg
  DESTINATION share/${PROJECT_NAME})

ament_package()
