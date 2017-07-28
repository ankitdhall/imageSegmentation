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
#include "header.h"
using namespace cv;
using namespace std;

Mat colSeg(Mat image, int winSize)//color segmentation only pair<Mat, vector<pair<pair<Point, Vec3b>, int > > >
{
    Mat gradient = generateGradient(image);
    Mat texture = generateTexture(gradient,winSize);

    printTexture(texture);
    segmentTexture(texture);
    //resize(image, image, Size(320,480));

    namedWindow( "Display window", 2);
    imshow( "Display window", image );

    int ROWS=image.rows,COLS=image.cols;//rows and cols of image
    RNG rng(12345);//random seed init

    Mat img(image.rows, image.cols, 0.0);
    //Mat mark(ROWS, COLS, CV_8UC1, 0.0);//visited mark
    Mat col(ROWS, COLS, CV_8UC1, 0.0);//
    Mat segm(ROWS, COLS, CV_8UC3, 0.0);

    medianBlur(image,image,5);//blur image
    cvtColor(image, img, CV_BGR2HSV);//change color space BGR to HSV

    int bins=40;//minimum segments
    int maxBins=100;//max segments, should be less than 250
    double markCount=0;//count number of pixels marked into one of the segments

    int i=0;//loop variable for bins
    double thresh=0.90;//thresh for pixels in bins

    vector<pair<pair<Point, Vec3b>, int> > Points;

    vector<Vec3b> avgColBGR;
    vector<int> pixelsInArea;
    Point pp;//randomly select a point to seed segment on

    unsigned long long avgVal[3]= {0,0,0}; //avgVal holds BGR color values
    int pixelCount=0;//pixel covered in a segment

    queue<Point> Q;//declare queue

    Point now;//hold neighbors
    Vec3b curr, next;//curr (for now) and next (for nos's neighbors) HSV vectors

    while(true)
    {
        if((markCount/(ROWS*COLS)>thresh && i>bins)  || i>maxBins)
        {
            break;
        }
        i++;

        pp.x=rng.uniform(0, COLS);//randomly get x co-ordinate
        pp.y=rng.uniform(0, ROWS);//randomly get y co-ordinate


        while(col.at<uchar>(pp.y, pp.x)!=0)//reselect point if already in one of the segments
        {
            pp.x=rng.uniform(0, COLS);
            pp.y=rng.uniform(0, ROWS);
        }

        Vec3b seed=img.at<Vec3b>(pp.y, pp.x);//HSV vector at (pp.x, pp.y)

        avgVal[0]=avgVal[1]=avgVal[2]=0;
        pixelCount=0;


        Q.push(pp);//push initial seed (pp.x, pp.y) onto queue
        col.at<uchar>(pp.y, pp.x)=i;//mark


        while(!Q.empty())//BFS
        {
            now=Q.front();//pop front of Q
            Q.pop();

            curr=img.at<Vec3b>(now.y, now.x);//HSV for pixel at (now.x,now.y)


            for(int p=-1; p<=1; p++)//looking for neighbors x-axis
            {
                for(int q=-1; q<=1; q++)//looking for neighbors y-axis
                {
                    if(0<=now.x+p && now.x+p<COLS && 0<=now.y+q && now.y+q<ROWS)//neighbor co-ord valid (in image)
                    {
                        next=img.at<Vec3b>(now.y+q, now.x+p);//HSV vector for neighbor
                        if(col.at<uchar>(now.y+q, now.x+p)==0 && (similar(next, seed) || farPtSimilar(next, curr)))
                        {
                            //if it's not marked already, check for similarity

                            Q.push(Point(now.x+p, now.y+q));//push point onto fringe
                            col.at<uchar>(now.y+q, now.x+p)=i;//mark the point on matrix
                            markCount+=1;//increase no. of pixels marked
                            segm.at<Vec3b>(now.y, now.x)=image.at<Vec3b>(pp.y, pp.x);//color the pixel with seed's color

                            for(int r=0; r<3; r++)
                            {
                                avgVal[r]+=(int)image.at<Vec3b>(now.y+q, now.x+p)[r];
                            }
                            pixelCount+=1;
                        }
                    }
                }
            }
        }

        if(pixelCount!=0)
        {
            for(int r=0; r<3; r++)
            {
                avgVal[r]/=pixelCount;
            }

            Vec3b vec=Vec3b(avgVal[0],avgVal[1],avgVal[2]);//return avgCol(B,G,R)

            pixelsInArea.push_back(pixelCount);
            avgColBGR.push_back(vec);

            Points.push_back(make_pair(make_pair(pp, vec),pixelCount));//push back point(pp.x, pp.y) and avgCol from that seed
        }

    }

    cout<<"Regions:"<<i<<",Perc:"<<100*markCount/(ROWS*COLS)<<endl;//display region count and segment percentage

    namedWindow("final",2);
    imshow("final",segm);
    //waitKey(0);
    //imwrite("colorSegment.jpg",segm);

    Mat combined = regionMerge(image,texture, col, segm, pixelsInArea, avgColBGR, winSize);
    //return make_pair(segm, Points);*/
    return combined;
}


bool similar(Vec3b now, Vec3b actual)//current and seed color distance metric for only color segmentation
{
    //20 60 60
    //15 30 70
    if(abs(now[0]-actual[0])<6)// && abs(now[1]-actual[1])<30 && abs(now[2]-actual[2])<70)
        return 1;
    return 0;
}


bool farPtSimilar(Vec3b next, Vec3b curr)//local color distance metric for only color segmentation
{
    //2 5 5
    //7 5 5
    if(abs(next[0]-curr[0])<=0)// && abs(next[1]-curr[1])<5 && abs(next[2]-curr[2])<5)
        return 1;
    return 0;
}


