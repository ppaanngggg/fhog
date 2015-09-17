#ifndef FHOG_H
#define FHOG_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>

namespace fhog {

	cv::Mat fhog( const cv::Mat_<float> &mximage, const int sbin = 4);

}

#endif
