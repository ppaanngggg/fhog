#include "fhog.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// #define HEIGHT 512
// #define WIDTH 512
// #define DEPTH 3

int main(){
    Mat image = imread("lena.jpg",0);

    Mat tmp_image;
    image.convertTo(tmp_image, CV_32FC1);
    cout<<image.rows<<" "<<image.cols<<endl;

    Mat fhog_feature = fhog::fhog(tmp_image);
    cout<<fhog_feature.rows<<" "<<fhog_feature.cols<<endl;
    vector<Mat> feats;
    split(fhog_feature, feats);
    for (size_t i = 0; i < feats.size(); i++) {
        cout<<feats[i]<<endl;
        imshow("feat", feats[i]);
        waitKey();
    }


    return 0;
}
