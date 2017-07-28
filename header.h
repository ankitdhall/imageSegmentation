/*
Authors:
Ankit Dhall and Yash Chandak
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include<math.h>
#include <queue>
#include <time.h>
#include<fstream>
using namespace cv;
using namespace std;

//functions from texture.cpp
int textureDifference(Vec4b p1, Vec4b p2);
int farTextureDifference(Vec4b p1, Vec4b p2);
void printTexture(Mat texture);
void printAtan();
Mat generateGradient(Mat img);//generates gradient for each pixel
Mat generateTexture(Mat gradient, int windowSize);//statistical texture pattern measure 20x20 patch
void segmentTexture(Mat texture);//segment image according to texture only

//functions from color.cpp
Mat colSeg(Mat image, int winSize);//color segmentation only
bool similar(Vec3b now, Vec3b actual);//current and seed color distance metric for only color segmentation
bool farPtSimilar(Vec3b next, Vec3b curr);//local color distance metric for only color segmentation

//
Mat regionMerge(Mat image ,Mat texture, Mat col, Mat segm, vector<int>pixelsInArea, vector<Vec3b>avgColBGR, int winSize);

//histogram
void hist(Mat);
