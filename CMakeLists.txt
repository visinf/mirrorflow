##### CMake entry point

cmake_minimum_required (VERSION 2.6)
project (mirrorFlow)

##### make release version
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++11)

# OpenCV
set( OPENCV_VER       2.4.13 )
set( OPENCV_PATH "/visinf/home/jhur/programming/lib/opencv-2.4.13-wo-cuda/install")
set( OPENCV_INC_DIR "${OPENCV_PATH}/include" )
set( OPENCV_LIB_DIR "${OPENCV_PATH}/lib" )
set( OPENCV_LIB   
	${OPENCV_LIB_DIR}/libopencv_core.so.${OPENCV_VER}
	${OPENCV_LIB_DIR}/libopencv_highgui.so.${OPENCV_VER}
	${OPENCV_LIB_DIR}/libopencv_imgproc.so.${OPENCV_VER}
	${OPENCV_LIB_DIR}/libopencv_calib3d.so.${OPENCV_VER}	
	${OPENCV_LIB_DIR}/libopencv_photo.so.${OPENCV_VER}
	${OPENCV_LIB_DIR}/libopencv_video.so.${OPENCV_VER}	
)


# PNG++
set( PNG_PP_INC_DIR "/visinf/home/jhur/programming/lib/png++" )
set( LIBPNG_INC_DIR "/visinf/home/jhur/programming/lib/libpng/libpng/install_re/include" )
set( LIBPNG_LIB_DIR "/visinf/home/jhur/programming/lib/libpng/libpng/install_re/lib" )
set( LIBPNG_LIB   
	libpng16.so
)


set( MAIN_INC_DIR "src")


########################## NO CHANGES BEYOND THIS POINT ##########################

##### include & link
  
# main project 
include_directories(
  ${MAIN_INC_DIR}
  ${OPENCV_INC_DIR}
  ${PNG_PP_INC_DIR}
  ${LIBPNG_INC_DIR}
)
link_directories(
  ${OPENCV_LIB_DIR}
  ${LIBPNG_LIB_DIR}  
)

##### project
FILE(GLOB MAIN_SRC_FILES "src/*.cpp" "src/*.h")
FILE(GLOB TPSEG_OPENCV_SRC_FILES "src/tpseg/*.cpp" "src/tpseg/*.h")
FILE(GLOB GCO_SRC_FILES "src/gco/*.cpp" "src/gco/*.h")
FILE(GLOB QPBO_SRC_FILES "src/qpbo/*.cpp" "src/qpbo/*.h")


##### build 
add_executable(mirrorFlow
  ${MAIN_SRC_FILES}
  ${TPSEG_OPENCV_SRC_FILES}   
  ${GCO_SRC_FILES}   
  ${QPBO_SRC_FILES}
)
target_link_libraries(mirrorFlow
  ${OPENCV_LIB}
  ${LIBPNG_LIB}
)
