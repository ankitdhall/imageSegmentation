/*TODO
    *generate common convolution function for all kernels
    *texture map generate can be sped up with integral image concept
    *size of texture kernel should be dependent on image size
    *convert texture's structure from Mat to 3d array after visualisation
    *parallelization
    *account for gradient value also along with direction
*/

//header files

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
    if(abs(now[0]-actual[0])<35 && abs(now[1]-actual[1])<60 && abs(now[2]-actual[2])<80)
        return 1;
    return 0;
}

bool farPtSimilar2(Vec3b next, Vec3b curr)//local color distance metric for final merge
{
    //2 5 5
    if(abs(next[0]-curr[0])<3 && abs(next[1]-curr[1])<5 && abs(next[2]-curr[2])<7)
        return 0;
    return 0;
}


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
    if(abs(p1[0]-p2[0])<100 &&  abs(p1[1]-p2[1])<40 && abs(p1[2]-p2[2])<100)// && abs(p1[3]-p2[3])<30)
        return 1;
    if(abs(p1[0]-p2[0])<30 &&  abs(p1[1]-p2[1])<15 && abs(p1[2]-p2[2])<30)// && abs(p1[3]-p2[3])<15)
        return 2;
    else
        return 0;

}
int farTextureDifference(Vec4b p1, Vec4b p2)//local texture distance metric only texture segment
{
    if(abs(p1[0]-p2[0])<20 &&  abs(p1[1]-p2[1])<5 && abs(p1[2]-p2[2])<20)// && abs(p1[3]-p2[3])<15)
        return 2;
    else
        return 0;

}

bool similar(Vec3b now, Vec3b actual)//current and seed color distance metric for only color segmentation
{
    //20 60 60
    if(abs(now[0]-actual[0])<15 && abs(now[1]-actual[1])<30 && abs(now[2]-actual[2])<70)
        return 1;
    return 0;
}

bool farPtSimilar(Vec3b next, Vec3b curr)//local color distance metric for only color segmentation
{
    //2 5 5
    if(abs(next[0]-curr[0])<7 && abs(next[1]-curr[1])<5 && abs(next[2]-curr[2])<5)
        return 1;
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
    namedWindow("gradient",2);
    imshow("gradient",gradient);

    namedWindow("gradientVal",2);
    imshow("gradientVal",gradVal);

    return gradient;
}

//window size optimal at 20 for most image sizes :)
Mat generateTexture(Mat gradient, int windowSize = 20)//statistical texture pattern measure 20x20 patch
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

    namedWindow("texture",2);
    imshow("texture",texture);
    imwrite("texture.png",texture);
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

    namedWindow("texture segment",2);
    imshow("texture segment",mark);
    imwrite("textureSegment.jpg",mark);
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
            if( 0.85 <= rat && rat <= 1.1)//check absolute size with threshold also
            {//cout<<"fa\n";
                return 0;
            }
            else
            {//cout<<"tr\n";
                return 1;
            }
        }
        else if(tSim && !cSim)
        {
            float rat = nextTotalPixels/currTotalPixels;
            if( 0.75 <= rat && rat <= 1.25)//check absolute size with threshold also
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

void regionMerge(Mat image ,Mat texture, Mat col, Mat segm, vector<int>pixelsInArea, vector<Vec3b>avgColBGR)
{//merge the regions considering both texture and color segmentation
    Mat combined(col.rows, col.cols, CV_8UC3, 0.0);
    int jump = 10;//windowSize/2 in generate texture
    namedWindow("segfinal",2);

    Point seed, curr, next;
    Mat marker(texture.rows, texture.cols, CV_8UC1, 0.0);
    queue<Point> Q;
    int segNo = 0;

    for(int i = 20; i<col.cols-20; i+=10)
    {
        for(int j = 20; j<col.rows-20; j+=10)
        {
            seed.x = i/10;
            seed.y = j/10;

            //cout<<"seed : "<<seed.x<<" "<<seed.y<<" "<<endl;

            Vec4b seedTexture = texture.at<Vec4b>(seed.y, seed.x);//removed -1
            Vec3b seedColor = segm.at<Vec3b>(seed.y*10, seed.x*10);//check if col!=0
            int seedSegment = (int)col.at<uchar>(seed.y*10, seed.x*10);
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
                Vec3b currColor = segm.at<Vec3b>(curr.y*10, curr.x*10);
                int currSegment = (int)col.at<uchar>(curr.y*10, curr.x*10);
                int currTotalPixels = pixelsInArea[currSegment-1];

                for(int p = -10; p<=10; p+=10)
                {
                    for(int q=-10; q<=10; q+=10)
                    {
                        next.x = curr.x + p/10;
                        next.y = curr.y + q/10;
                        if(0<=next.x && next.x<texture.cols && 0<=next.y && next.y<texture.rows
                                && marker.at<uchar>(next.y, next.x)==0)
                        {



                        //cout<<next.y<< " "<<next.x<<endl;
                        Vec4b nextTexture = texture.at<Vec4b>(next.y, next.x);
                        Vec3b nextColor = segm.at<Vec3b>(next.y*10, next.x*10);
                        int nextSegment = (int)col.at<uchar>(next.y*10, next.x*10);
                        int nextTotalPixels = pixelsInArea[nextSegment-1];

                        //check if not zero marker then proceed
                        int choice=ifSimilar(nextTexture, seedTexture, currTexture,
                                             nextColor, seedColor, currColor,
                                             nextSegment,seedSegment, currSegment,
                                             nextTotalPixels,seedTotalPixels, currTotalPixels);

                        if(choice==1)//then merge both and color with updated color
                        {
                            //cout<<"push : "<<next.y<<" "<<next.x<<" "<<segNo<<endl;
                            Q.push(next);//push the next pixel
                            marker.at<uchar>(next.y, next.x) = 2; //mark the next pixel
                            //color the block
                            //check for " <= "
                            for(int g=-10; g<=10; g++)
                            {
                                for(int h=-10; h<=10; h++)
                                {
                                    if(0<=(next.y*10+h) && (next.y*10+h)<combined.rows
                                            && 0<=(next.x*10+g) && (next.x*10+g)<combined.cols)
                                            {
                                                combined.at<Vec3b>(next.y*10+h, next.x*10+g)=image.at<Vec3b>(seed.y*10, seed.x*10);//combined.at<Vec3b>(j+q, i+p);
                                            }
                                }
                            }
                        }
                       }

                    }
                }


            }
            imshow("segfinal",combined);
                //waitKey(2);
        }
    }
    imshow("segfinal",combined);
    waitKey(0);
    imwrite("finalSegment.jpg",combined);
}

pair<Mat, vector<pair<pair<Point, Vec3b>, int > > > colSeg(Mat image)//color segmentation only
{
    Mat gradient = generateGradient(image);
    Mat texture = generateTexture(gradient);

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
    imwrite("colorSegment.jpg",segm);

    regionMerge(image,texture, col, segm, pixelsInArea, avgColBGR);
    return make_pair(segm, Points);

}



int main()
{
    Mat image= imread("pag.jpg",1);
    imwrite("originalImage.jpg",image);


    //resize(image, image, Size(320,480));
    namedWindow("image",2);
    imshow("image",image);

    pair<Mat, vector<pair<pair<Point, Vec3b> ,int > > >retVal= colSeg(image);
    waitKey(0);
    return 0;
}

