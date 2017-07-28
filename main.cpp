/*
Authors:
Ankit Dhall and Yash Chandak
*/
/*TODO
    *generate common convolution function for all kernels
    *texture map generate can be sped up with integral image concept
    *size of texture kernel should be dependent on image size
    *convert texture's structure from Mat to 3d array after visualisation
    *parallelization
    *account for gradient value also along with direction
*/

//header files
#include <opencv2/photo/photo.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <queue>
#include <time.h>
#include <fstream>
#include "header.h"
using namespace cv;
using namespace std;

int textureDifference2(Vec4b p1, Vec4b p2)//current and seed texture distance metric for final merge
{
    //interchanged 1 and 2 return statements
    if(abs(p1[0]-p2[0])<60 &&  abs(p1[1]-p2[1])<30 && abs(p1[2]-p2[2])<60)// && abs(p1[3]-p2[3])<30)
        return 1;
    else
        return 0;

}
int farTextureDifference2(Vec4b p1, Vec4b p2)//local texture distance metric for final merge
{
    if(abs(p1[0]-p2[0])<10 &&  abs(p1[1]-p2[1])<5 && abs(p1[2]-p2[2])<10)// && abs(p1[3]-p2[3])<15)
        return 0;
    else
        return 0;

}

bool similar2(Vec3b now, Vec3b actual)//current and seed color distance metric for final merge
{
    //20 60 60
    //35 60 60
    if(abs(now[0]-actual[0])<3)// && abs(now[1]-actual[1])<60 && abs(now[2]-actual[2])<80)
        return 1;
    return 0;
}

bool farPtSimilar2(Vec3b next, Vec3b curr)//local color distance metric for final merge
{
    //2 5 5
    if(abs(next[0]-curr[0])<1)// && abs(next[1]-curr[1])<5 && abs(next[2]-curr[2])<7)
        return 0;
    return 0;
}



int ifSimilar(Vec4b nextTexture,Vec4b seedTexture, Vec4b currTexture,
              Vec3b nextColor, Vec3b seedColor, Vec3b currColor,
              int nextSegment, int seedSegment, int currSegment,
              int nextTotalPixels, int seedTotalPixels, int currTotalPixels)//distance metric score for final merge(change binary score to continuous valued)
{
    if(nextSegment == 0 && nextTexture[1] >= 1)
    {
        return textureDifference2(nextTexture, seedTexture);
    }
    else if(nextSegment != 0 && nextTexture[1] < 2)
    {
        return similar2(nextColor, seedColor);
    }
    else if(nextSegment != 0 && nextTexture[1] >= 1)
    {
        int cSim =0, tSim =0;
        if(similar2(nextColor, seedColor) || farPtSimilar2(nextColor, currColor))
            cSim = 1;
        if(textureDifference2(nextTexture, seedTexture) || farTextureDifference(nextTexture, currTexture))
            tSim = 1;

        if(tSim && cSim)
        {
            return 1;
        }
        else if(!tSim && cSim)
        {
            float rat = nextTotalPixels/currTotalPixels;
            if( 0.85 <= rat && rat <= 1.5 && (nextTotalPixels > 50 || currTotalPixels > 50))//check absolute size with threshold also
            {
                //cout<<"fa\n";
                return 0;
            }
            else
            {
                //cout<<"tr\n";
                return 3;//return 1;
            }
        }
        else if(tSim && !cSim)
        {
            float rat = nextTotalPixels/currTotalPixels;
            if( 0.60 <= rat && rat <= 1.5 )//&& (nextTotalPixels > 50 || currTotalPixels > 50))//check absolute size with threshold also
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }

    }
    else
    {
        return 1;//consider merge
    }
}

Mat regionMerge(Mat image ,Mat texture, Mat col, Mat segm, vector<int>pixelsInArea, vector<Vec3b>avgColBGR, int winSize)
{
    //merge the regions considering both texture and color segmentation
    Mat combined(col.rows, col.cols, CV_8UC3, 0.0);
    int jump = winSize/2;//windowSize/2 in generate texture
    namedWindow("segfinal",2);


    //trial blur
    //medianBlur(image, image, 3);
    medianBlur(segm, segm, 7);
    //medianBlur(texture, texture, 3);
    ///
    /*namedWindow("1",2);
    namedWindow("2",2);
    namedWindow("3",2);
    imshow("1", image);
    imshow("2", segm);
    imshow("3", texture);*/
    ///


    Point seed, curr, next;
    Mat marker(texture.rows, texture.cols, CV_8UC1, 0.0);
    queue<Point> Q;
    int segNo = 0;

    short dy,dx;

    for(int i = winSize; i<col.cols-winSize; i+=jump)
    {
        for(int j = winSize; j<col.rows-winSize; j+=jump)
        {
            seed.x = i/jump;
            seed.y = j/jump;

            //cout<<"seed : "<<seed.x<<" "<<seed.y<<" "<<endl;

            Vec4b seedTexture = texture.at<Vec4b>(seed.y, seed.x);//removed -1
            Vec3b seedColor = segm.at<Vec3b>(seed.y*jump, seed.x*jump);//check if col!=0
            int seedSegment = (int)col.at<uchar>(seed.y*jump, seed.x*jump);
            int seedTotalPixels = pixelsInArea[seedSegment-1];

            if((seedSegment == 0 && seedTexture[1] < 4 )|| marker.at<uchar>(seed.y, seed.x))
            {
                //cout<<"here:\n";
                continue;
            }

            segNo++;
            Q.push(seed);
            marker.at<uchar>(seed.y, seed.x)=segNo;

            while(!Q.empty())
            {
                curr = Q.front();
                Q.pop();

                Vec4b currTexture = texture.at<Vec4b>(curr.y, curr.x);
                Vec3b currColor = segm.at<Vec3b>(curr.y*jump, curr.x*jump);
                int currSegment = (int)col.at<uchar>(curr.y*jump, curr.x*jump);
                int currTotalPixels = pixelsInArea[currSegment-1];


                for(int p =-jump; p<=jump; p+=jump)
                {
                    for(int q=-jump; q<=jump; q+=jump)
                    {
                        next.x = curr.x + p/jump;
                        next.y = curr.y + q/jump;
                        if(0<=next.x && next.x<texture.cols && 0<=next.y && next.y<texture.rows
                                && marker.at<uchar>(next.y, next.x)==0)
                        {


                            //cout<<next.y<< " "<<next.x<<endl;
                            Vec4b nextTexture = texture.at<Vec4b>(next.y, next.x);
                            Vec3b nextColor = segm.at<Vec3b>(next.y*jump, next.x*jump);
                            int nextSegment = (int)col.at<uchar>(next.y*jump, next.x*jump);
                            int nextTotalPixels = pixelsInArea[nextSegment-1];

                            //check if not zero marker then proceed
                            int choice=ifSimilar(nextTexture, seedTexture, currTexture,
                                                 nextColor, seedColor, currColor,
                                                 nextSegment,seedSegment, currSegment,
                                                 nextTotalPixels,seedTotalPixels, currTotalPixels);

                            if(choice)//then merge both and color with updated color
                            {
                                //cout<<"push : "<<next.y<<" "<<next.x<<" "<<segNo<<endl;
                                Q.push(next);//push the next pixel
                                marker.at<uchar>(next.y, next.x) = 2; //mark the next pixel
                                //color the block
                                //check for " <= "
                                if(choice == 3)
                                {
                                    dx = -1 + (rand() % 3);
                                    dy = -1 + (rand() % 3);
                                    //cout<<"3";
                                }

                                for(int g=-jump; g<=jump; g++)
                                {
                                    for(int h=-jump; h<=jump; h++)
                                    {
                                        if(0<=(next.y*jump+h) && (next.y*jump+h)<combined.rows
                                                && 0<=(next.x*jump+g) && (next.x*jump+g)<combined.cols)
                                        {
                                            if(choice == 3)
                                            {
                                                if((next.y + dy) < combined.rows && (next.x + dx) < combined.cols
                                                   && (next.y + dy) >= 0 && (next.x + dx) >= 0)
                                                combined.at<Vec3b>(next.y*jump+h, next.x*jump+g)=combined.at<Vec3b>(next.y*jump + dy, next.x*jump + dx);
                                                else
                                                combined.at<Vec3b>(next.y*jump+h, next.x*jump+g)=image.at<Vec3b>(seed.y*jump, seed.x*jump);
                                                //cout<<next.y*jump+h<<","<< next.x*jump+g<<",,"<<next.y + dy<<","<<next.x+ dx<<endl;
                                            }
                                            else
                                            combined.at<Vec3b>(next.y*jump+h, next.x*jump+g)=image.at<Vec3b>(seed.y*jump, seed.x*jump);//combined.at<Vec3b>(j+q, i+p);

                                        }
                                    }
                                }
                            }
                        }

                    }
                }



            }
            //imshow("segfinal",combined);
            //waitKey(2);
        }
    }

    ///imshow("box",image);
    imshow("segfinal",combined);
    //resize(combined, combined, Size(320, 480));
    //namedWindow("hehe", 2);
    medianBlur(combined, combined, 11);
    //imshow("hehe", combined);
    waitKey(0);
    //imwrite("finalSegment.jpg",combined);
    return combined;
}


void crop(Mat combined, Mat original)
{
    //namedWindow("Smoothcombined", 2);
    //imshow("Smoothcombined", combined);

    namedWindow("finalll", 2);
    ///Mat original=combined;
    imshow("try",original);

    int ROWS = combined.rows;
    int COLS = combined.cols;

    Mat marker(ROWS, COLS, CV_8UC1, 0.0);

    Point seed, curr, next;

    int segNo = 0;

    short dy,dx;

    int minx, maxx, miny, maxy;

    queue<Point> Q;
    cvtColor(combined, combined, CV_BGR2HSV);
    ///convert the image to HSV space
    for(int i = 0; i<COLS; ++i)
    {
        for(int j = 0; j<ROWS; ++j)
        {
            seed.x = i;
            seed.y = j;
            minx=maxx=seed.x;
            miny=maxy=seed.y;
            //cout<<"seed : "<<seed.x<<" "<<seed.y<<" "<<endl;

            Vec3b seedColor = combined.at<Vec3b>(seed.y, seed.x);//check if col!=0
            //cout<<seed.x<<","<<seed.y<<endl;
            //cout<<marker.at<uchar>(seed.y, seed.x)<<" ";
            if(marker.at<uchar>(seed.y, seed.x) != 0)
            {
                //cout<<"here:\n";
                continue;
            }

            segNo++;
            Q.push(seed);
            marker.at<uchar>(seed.y, seed.x)=segNo;

            while(!Q.empty())
            {
                curr = Q.front();
                Q.pop();

                Vec3b currColor = combined.at<Vec3b>(curr.y, curr.x);

                for(int p = -1; p<=1; ++p)
                {
                    for(int q=-1; q<=1; ++q)
                    {
                        next.x = curr.x + p;
                        next.y = curr.y + q;
                        if(0<=next.x && next.x<COLS && 0<=next.y && next.y<ROWS
                                && marker.at<uchar>(next.y, next.x)==0)
                        {


                            //cout<<next.y<< " "<<next.x<<endl;
                            Vec3b nextColor = combined.at<Vec3b>(next.y, next.x);

                            if(abs(nextColor[0]-seedColor[0]) <= 2)//choice)//then merge both and color with updated color
                            {
                                //cout<<"push : "<<next.y<<" "<<next.x<<" "<<segNo<<endl;
                                Q.push(next);//push the next pixel
                                marker.at<uchar>(next.y, next.x) = 2;//segNo; //mark the next pixel
                                //color the block
                                //check for " <= "
                                if(miny > next.y)
                                    miny=next.y;
                                if(maxy < next.y)
                                    maxy=next.y;
                                if(maxx < next.x)
                                    maxx=next.x;
                                if(minx > next.x)
                                    minx=next.x;
                            }
                        }//cout<<maxx<<","<<minx<<","<<maxy<<","<<miny<<endl;
                    }
                }
            }
            if(abs((maxx-minx)*(maxy-miny)) > 10000 && abs(maxx-minx)>50 && abs(maxy-miny)>50)
            {
                rectangle(original, Point(minx, maxy), Point(maxy, miny), Scalar(128, 255, 0), 5);
                waitKey(0);
                imshow("finalll", original);
            }
        }
    }
    imshow("finalll", original);
    waitKey(0);
}

int main()
{
    //Mat image= imread("C:/Users/student/Desktop/DIP/Sample Pictures/cr.jpg",1);
    Mat image= imread("a.jpg",1);
    hist(image);
    //imwrite("originalImage.jpg",image);


    //resize(image, image, Size(320,480));

    ///namedWindow("image",2);
    ///imshow("image",image);

    int winSize=20;
    //pair<Mat, vector<pair<pair<Point, Vec3b> ,int > > >retVal=
    Mat combined = colSeg(image, winSize);
    crop(combined, image);

    waitKey(0);
    return 0;
}

