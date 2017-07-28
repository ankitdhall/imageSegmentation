/*
Authors:
Ankit Dhall and Yash Chandak
*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "header.h"

using namespace std;
using namespace cv;
void hist(Mat src)
{
    Mat dst;

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> hsv_planes;
    split( src, hsv_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true;
    bool accumulate = false;

    Mat h_hist;

    /// Compute the histograms:
    calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );


    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    float sum=0,mu1=0,mu2=0;
    for(int i=0;i<h_hist.rows;i++)
    {
        cout<<h_hist.at<float>(i)<<endl;
        sum+=h_hist.at<float>(i);
    }
    float avg=(float)sum/h_hist.rows;
    cout<<avg<<endl;

    int peaks=0;
    sum=0;
    /// Draw for each channel
    for( int i = 1; i < histSize -1; i++ )
    {
        if(h_hist.at<float>(i)>=avg)
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
            if((h_hist.at<float>(i-1) < h_hist.at<float>(i)) && (h_hist.at<float>(i) > h_hist.at<float>(i+1)))
            {
                ++peaks;
                mu2+=(h_hist.at<float>(i)*i*i);
                mu1+=h_hist.at<float>(i)*i;
                sum+=h_hist.at<float>(i);

            }

        }

    }
    mu1/=sum;
    mu2/=sum;
    cout<<mu2<<","<<mu1<<endl;
    float var=mu2-(mu1*mu1);
    var/=100;
    /// Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );

    cout<<"Var:"<<var<<endl;
    cout<<"Peaks:"<<peaks<<endl;

}
