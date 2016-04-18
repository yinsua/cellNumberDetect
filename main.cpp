/*

          __   _,--="=--,_   __
         /  \."    .-.    "./  \
        /  ,/  _   : :   _  \/` \
        \  `| /o\  :_:  /o\ |\__/
         `-'| :="~` _ `~"=: |
            \`     (_)     `/
     .-"-.   \      |      /   .-"-.
.---{     }--|  /,.-'-.,\  |--{     }---.
 )  (_)_)_)  \_/`~-===-~`\_/  (_(_(_)  (
(       cell liquid color judge         )
 )               main.cpp              (
'---------------------------------------'

*/

#include <iostream>
#include <sstream>
#include <string>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <stdio.h>

using namespace cv;
using namespace std;

// global var ------->
Mat gray;
Mat src;
int edgeThresh = 220;
int holeThresh = 0;
int thresh = 100;
int maxElementPos = 10;
int ElementPosThresh = 10;
int ElementSharp = MORPH_RECT;
// <-----------------

void cvText(cv::Mat& img, const char* text, int x, int y)  
{  
    CvFont font; 
    double hscale = 0.6;  
    double vscale = 0.6;
    int linewidth = 2;  
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC,hscale,vscale,0,linewidth);  
    CvScalar textColor =cvScalar(255,255,0);
    CvPoint textPos =cvPoint(x, y);  
 
    IplImage s ;
    s = img;
    cvPutText(&s, text, textPos, &font,textColor);
} 

void showNumber(cv::Mat& img, int num, int x, int y){
    stringstream s;
    char a[10];
    s << num;
    s >> a;
	cvText(img,a,x,y);
}

/******************/
/* edge detection */
/******************/
void cvEdgeDetection(Mat bImage){
    Mat dst1,dst2,dst3;
    int x = ElementPosThresh - maxElementPos; //结构元素(内核矩阵)的尺寸
    int g_nStructElementSize = x >0 ? x : -x;
    //获取自定义核
    Mat element =getStructuringElement(ElementSharp,
        Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),
        Point(g_nStructElementSize, g_nStructElementSize ));
    if(x<0){
        erode(src,dst1,element);
        morphologyEx(src,dst2,MORPH_OPEN,element);
        morphologyEx(src,dst3,MORPH_TOPHAT,element);
    }
    else{
        dilate(src,dst1,element);
        morphologyEx(src,dst2,MORPH_CLOSE,element);
        morphologyEx(src,dst3,MORPH_BLACKHAT,element);
    }
    //imshow("Erode/Dilate",dst1);
    //imshow("Open/Close",  dst2);
    imshow("TOPHAT/BLACKHAT",  dst3);

    Mat gray1,gray2,gray3;

    cvtColor(dst1,gray1,COLOR_RGB2GRAY);
    cvtColor(dst2,gray2,COLOR_RGB2GRAY);
    cvtColor(dst3,gray3,COLOR_RGB2GRAY);

    blur(gray1,gray1,Size(3,3));
    blur(gray2,gray3,Size(3,3));
    blur(gray3,gray3,Size(3,3));

    int _width  = src.cols;
    int _height = src.rows;
    //cout << src.depth() ;
    IplImage* img1 = cvCreateImage(cvSize(_width,_height),IPL_DEPTH_8U,1);
    IplImage* img2 = cvCreateImage(cvSize(_width,_height),IPL_DEPTH_8U,1);
    IplImage* img3 = cvCreateImage(cvSize(_width,_height),IPL_DEPTH_8U,1);

    IplImage tmpImage1;
    tmpImage1 = gray1;

    IplImage tmpImage2;
    tmpImage2 = gray2;

    IplImage tmpImage3;
    tmpImage3 = gray3;

    cvNot(&tmpImage3,&tmpImage3);

    cvShowImage("reverse",&tmpImage3);

    cvThreshold(&tmpImage1,img1,thresh,255,CV_THRESH_BINARY);
    cvThreshold(&tmpImage2,img2,thresh,255,CV_THRESH_BINARY);
    cvThreshold(&tmpImage3,img3,thresh,255,CV_THRESH_BINARY);

    /*
    cvAdaptiveThreshold(&tmpImage1,img1,255);
    cvAdaptiveThreshold(&tmpImage2,img2,255);
    cvAdaptiveThreshold(&tmpImage3,img3,255);
    */
    
    //cvShowImage("Erode/Dilate Threshold",img1);
    //cvShowImage("Open/Close Threshold",  img2);
    cvShowImage("TOPHAT/BLACKHAT Threshold",  img3);

    vector < vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(Mat(img3),contours,hierarchy,RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    Mat cImage = Mat::zeros(bImage.size(),CV_8UC3);
    Mat dImage = Mat::zeros(bImage.size(),CV_8UC3);
    Mat eImage = Mat::zeros(bImage.size(),CV_8UC3);
    dImage = src.clone();

	int number = 0;

	//for(size_t i=0; i < contours.size(); i++){
    for(int i=0; i>=0; i=hierarchy[i][0]){
		size_t count = contours[i].size();
		if( count < 6 )
			continue;
		Mat pointsf;
		Mat(contours[i]).convertTo(pointsf, CV_32F);
		RotatedRect box = fitEllipse(pointsf);

		if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*10)
			continue;
		if( MAX(box.size.width, box.size.height) > MAX(cImage.rows,cImage.cols)/2)
			continue;
		if( MIN(box.size.width, box.size.height) <= holeThresh)
			continue;

        Scalar color(rand()&255, rand()&255,rand()&255);
		//drawContours(dImage, contours, i, Scalar(255,0,0), 0, 8, hierarchy);

        //vector < vector<cv::Point> > approx;
        //approxPolyDP(contours,approx,10.0,true);
		//drawContours(eImage, approx, i, color, CV_FILLED, 8, hierarchy);

		ellipse(cImage, box, Scalar(0,0,255), 1);
		ellipse(cImage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1);

		ellipse(dImage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(255,0,0), 1);

		Point2f vtx[4];
		box.points(vtx);
		for( int j = 0; j < 4; j++ ){
			//line(cImage, vtx[j], vtx[(j+1)%4], cv::Scalar(0,255,0), 1);
		}
        number++;
	}	
    cvText(dImage,"Num:",dImage.cols-90,dImage.rows-9);
    showNumber(dImage,number,dImage.cols-40,dImage.rows-9);

	//imshow("cImage",cImage);
	imshow("dImage",dImage);
	//imshow("eImage",eImage);
	//return maxValue;
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&img3);

}

// threshold trackbar callback
static void onTrackbar( int, void* ) {
    
    Mat edge = gray >= edgeThresh;
    GaussianBlur(edge,edge,Size(5,5),0,0);

    //imshow("Distance Map", edge);

    cvEdgeDetection(edge);
}

const char* keys =
{
    "{1| |cell.jpg|input image file}"
};

int main( int argc, const char** argv ){
    //CommandLineParser parser(argc, argv, keys);
    //string filename = parser.get<string>("1");
    //src = imread(filename.c_str());
    src = imread("cell.jpg");
    cvtColor(src,gray,CV_RGB2GRAY);
    
    
    if(gray.empty())
    {
        //printf("Cannot read image file: %s\n", filename.c_str());
        printf("Cannot read image file ");
        return -1;
    }

    namedWindow("Distance Map", 1);
    createTrackbar("Brightness Threshold", "Distance Map", &edgeThresh, 255, onTrackbar, 0);
    createTrackbar("Hole Threshold", "Distance Map", &holeThresh, 255, onTrackbar, 0);
    createTrackbar("ElementPos Threshold", "Distance Map", &ElementPosThresh, 64, onTrackbar, 0);
    createTrackbar("Threshold", "Distance Map", &thresh, 255, onTrackbar, 0);

    for(;;)
    {
        // Call to update the view
        onTrackbar(0, 0);

        int c = waitKey(0) ;

        if( c == 27 )
            break;
        ElementSharp = 
            c == 'e' ? MORPH_ELLIPSE :
            c == 'r' ? MORPH_RECT :
            c == 'c' ? MORPH_CROSS : MORPH_RECT;
    }
}


