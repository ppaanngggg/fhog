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

    Mat fhog_feature = fhog(image);
    vector<Mat> feats;
    split(fhog_feature, feats);
    for (size_t i = 0; i < feats.size(); i++) {
        cout<<feats[i]<<endl;
        imshow("feat", feats[i]);
        waitKey();
    }


    return 0;
}
