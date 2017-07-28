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

int textureDifference(Vec4b p1, Vec4b p2)//current and seed texture distance metric only texture segment
{
    /*
      0 - mean :: 1 - total :: 2 - variance :: 3 - mode

      mode same point 2
      mode adjacent 1
      otherwise 0

      variance should be very similar

      mean very similar

      density should be very similar
    */

    //interchanged 1 and 2 return statements
    //50 30 15
    if(abs(p1[0]-p2[0])<50 &&  abs(p1[1]-p2[1])<30 && abs(p1[2]-p2[2])<15)// && abs(p1[3]-p2[3])<30)
        return 1;
    if(abs(p1[0]-p2[0])<25 &&  abs(p1[1]-p2[1])<15 && abs(p1[2]-p2[2])<30)// && abs(p1[3]-p2[3])<15)
        return 2;
    else
        return 0;

}


int farTextureDifference(Vec4b p1, Vec4b p2)//local texture distance metric only texture segment
{
    //30 20 10
    if(abs(p1[0]-p2[0])<30 &&  abs(p1[1]-p2[1])<20 && abs(p1[2]-p2[2])<10)// && abs(p1[3]-p2[3])<15)
        return 2;
    else
        return 0;

}


void printTexture(Mat texture)//file write 4d texture feature
{
    imwrite("texture.png",texture);
    fstream out;
    out.open("texture.txt", ios::out);

    for(int i =0; i<texture.rows; i++)
    {
        for(int j = 0; j<texture.cols; j++)
        {
            for(int k=0; k<4; k++)
            {
                out<<int(texture.at<Vec4b>(i,j)[k])<<" ";
            }
            out<<"\t\t";
        }
        out<<"\n";
    }
    out.close();
}

void printAtan()//prints atan (for verification)
{
    int min=500, max=-500, val;
    for(int dx = -180; dx< 180; dx+=30)
    {
        for(int dy = -180; dy<180; dy+=30)
        {
            //val = (int)(atan2(dy,dx)*180.0/3.14);
            val = (((int((atan2(dy,dx)*180.0/3.14)) + 180)%180)/45 );
            if (val>max)
                max =val;
            if(val<min)
                min = val;
            cout<<val<<" ";
        }
        cout<<"\n\n";
    }
    cout<<"max : "<<max<<" min : "<<min<<endl;
}

Mat generateGradient(Mat img)//generates gradient for each pixel
{
    Mat image(img.rows, img.cols, CV_8UC1,0.0);
    cvtColor(img, image,CV_RGB2GRAY);
    Mat gradient(image.rows, image.cols, CV_8UC1, 255.0);
    Mat gradVal(img.rows, img.cols, CV_8UC1,0.0);

    //KernelSize should be odd number
    int kernelSize = 3;
    int kS = kernelSize/2;
    int kernelX[][3] = { {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };

    int kernelY[][3] = { {-1,-2,-1},
        {0,0,0},
        {1,2,1}
    };


    int dy, dx, slope, val;
    int thresh = 15;
    int darkness = 30;
    int temp;
    for(int i = kS; i< image.rows - kS; i++)
    {
        for (int j = kS; j<image.cols - kS; j++)
        {
            dx =0;
            dy =0;
            slope =0;

            for(int k = -kS; k<=kS; k++)
            {
                for(int l = -kS ; l<=kS; l++)
                {
                    dx += kernelX[kS + k][kS + l]*image.at<uchar>(i+l,j+k);
                    dy += kernelY[kS + k][kS + l]*image.at<uchar>(i+l,j+k);
                }

            }

            val = (abs(dx)+abs(dy));
            gradVal.at<uchar>(i,j) = val;
            if(val > thresh )
                gradient.at<uchar>(i,j) = (((int((atan2(dy,dx)*180.0/3.14)) + 180)%180)/45 );

        }
    }
    ///namedWindow("gradient",2);
    ///imshow("gradient",gradient);

    ///namedWindow("gradientVal",2);
    ///imshow("gradientVal",gradVal);

    return gradient;
}

//window size optimal at 20 for most image sizes :)
Mat generateTexture(Mat gradient, int windowSize)//statistical texture pattern measure 20x20 patch
{
    int jump = windowSize/2;
    Mat texture(gradient.rows/jump, gradient.cols/jump, CV_8UC4, 0.0);
    int mode,m;
    float mean, variance, probability[4], total;
    int r=0,c, orientation[4], dir;

    for(int i = jump; i<(texture.rows-1)*jump; i+=jump )
    {
        c=0;
        for(int j = jump; j<(texture.cols-1)*jump; j+=jump)
        {
            mean=0;
            variance=0;
            total=0, mode= 0;

            for(m =0; m<4; m++)
            {
                orientation[m]=0;
                probability[m]=0;
            }

            for(int k = -jump; k<jump; k++)
            {
                for(int l = -jump; l<jump; l++)
                {
                    dir = (int)gradient.at<uchar>(i+k,j+l);
                    if(dir!=255)
                    {
                        orientation[dir]++;
                        total++;
                    }
                }
            }

            if(total!=0)
            {
                for(m=0; m<4; m++)
                {

                    probability[m]= orientation[m]/total;
                    mean += probability[m]*m;

                    if(orientation[mode]<=orientation[m])
                        mode = m;
                }

                for(m=0; m<4; m++)
                    variance += ((m-mean)*(m-mean))*probability[m];

                total /=2;          //total can be maximum 4*jump^2
                mean *= 70;         //mapping mean value from 0-3 to 0-210
                variance *= 100;    //variance <= c^2/4 | c=3 here
                //mode *= 70;
                mode = 255;//50 + mode*10;

                texture.at<Vec4b>(r,c)[0] = (int)mean;
                texture.at<Vec4b>(r,c)[1] = (int)total;
                texture.at<Vec4b>(r,c)[2] = (int)variance;
                texture.at<Vec4b>(r,c)[3] = mode;

            }
            c++;
        }
        r++;
    }

    ///namedWindow("texture",2);
    ///imshow("texture",texture);
    //imwrite("texture.png",texture);
    return texture;
}



void segmentTexture(Mat texture)//segment image according to texture only
{
    Mat regions(texture.rows, texture.cols, CV_8UC1, 0.0);
    Mat mark(texture.rows, texture.cols, CV_8UC1, 0.0);

    int min_regions = 10;
    RNG rng(25);

    Point seedPoint, now;
    Vec4b curr, next, seed;
    queue<Point> Q;

    vector<pair<pair<Point, Vec3b>, int> > Points;
    int pixelCount=0;//pixel covered in a segment

    for(int i =250; i>10; i-=250/min_regions)
    {
        rng(12345);

        do
        {
            seedPoint.x=rng.uniform(0, texture.cols);
            seedPoint.y=rng.uniform(0, texture.rows);
            seed=texture.at<Vec4b>(seedPoint.y, seedPoint.x);
        }
        while(mark.at<uchar>(seedPoint.y, seedPoint.x) !=0 );

        mark.at<uchar>(seedPoint.y, seedPoint.x) = i;
        Q.push(seedPoint);

        while(!Q.empty())
        {
            now=Q.front();
            Q.pop();

            curr=texture.at<Vec4b>(now.y, now.x);

            for(int p=-1; p<=1; p++)
            {
                for(int q=-1; q<=1; q++)
                {
                    if(0<=now.x+p && now.x+p<texture.cols && 0<=now.y+q && now.y+q<texture.rows)
                    {
                        next=texture.at<Vec4b>(now.y+q, now.x+p);
                        if(mark.at<uchar>(now.y+q, now.x+p)==0 &&
                                (textureDifference(next, seed)==1 || farTextureDifference(next, curr)==2))
                        {
                            Q.push(Point(now.x+p, now.y+q));
                            mark.at<uchar>(now.y+q, now.x+p)=i;
                            //segm.at<Vec3b>(now.y, now.x)=image.at<Vec3b>(seedPoint.y, seedPoint.x);
                        }
                    }
                }
            }
        }
    }

    ///namedWindow("texture segment",2);
    ///imshow("texture segment",mark);
    //imwrite("textureSegment.jpg",mark);
}

