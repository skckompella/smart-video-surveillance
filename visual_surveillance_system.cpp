#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

#include<unistd.h>
#include<sys/socket.h>
#include<sys/types.h>
#include<netinet/in.h>
#include<arpa/inet.h>

#include "cvaux.h"
#include "cxmisc.h"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "colorhistogram.h"
//#include "people_stream.h"

#define DATASET_PATH  "/home/chaitanya/dip_project/dataset/S1-T1-C/video/pets2006/S1-T1-C/1/S1-T1-C."
#define TRAINING_PATH "/home/chaitanya/dip_project/third_phase_with_message_board/data/"
#define HIST_DATA_PATH "/home/chaitanya/dip_project/third_phase_with_message_board/hist.xml"

#define TRAINING_DATA_SIZE 1085

#define STATIONARY_THRESHOLD 5 //Number frames to check before considering an object to be stationary

#define LOCALHOST "127.0.0.1"  // Specifies localhost address so as to connect to a message board on the same system

#define TEST 1
#define STATIONARY 2
#define BAG 3
#define PERSON 4

using namespace cv;
using namespace std;


RNG rng(12345); //Random Number Generator

int send_msg(int index); //Function to send message to message board via socket

bool people_detector(Mat); //Function to detect if a blob is that of a person

bool bag_detector(Mat);


class Reference
{
public:


    Mat hist;
    bool is_person;
    bool is_bag;

    int keypoints_count;
    int corner_count;
    int compactness;
    int circle_count;
    double  h_w_ratio;


    Reference()
    {
        corner_count = 0;
        compactness = 0;
        circle_count = 0;
        h_w_ratio = 0;

        is_person=0;
        is_bag = 1;


    }



    void get_features(char fname[])
    {
        Mat src,gray;
        ColorHistogram c1;
        src = imread(fname,CV_LOAD_IMAGE_COLOR);
        if(! src.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return;
        }

        hist = c1.getabHistogram(src);
        normalize(hist,hist,0,hist.rows,NORM_MINMAX,-1,Mat());
        //cout<<"\n "<<hist<<endl<<endl;

        h_w_ratio =(double)src.rows/src.cols;
        //cout<<"\n H/W: "<<h_w_ratio;

    /*    cvtColor(src, gray, CV_BGR2GRAY);
        GaussianBlur( gray, gray, Size(9, 9), 2, 2 );

        vector<KeyPoint> keypoints;
        FAST(gray,keypoints, 9);
        keypoints_count = keypoints.size();


        vector<Point2f> corners;
        goodFeaturesToTrack(gray, corners, 5, .6, 3);
        corner_count = corners.size();
        //cout<<"\n Corners: "<<corner_count;


        vector<Vec3f> circles;
        HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, gray.rows/4, 200, 100 );
        circle_count = circles.size();
        //cout<<"\n Circles: "<<circle_count<<endl;*/

    }

};


//Blobs class contains the all data relating to a Blob in the current frame
class Blobs
{
public:
    int blob_id;
    Mat region;
    Point2f centroid;
    Mat hist;
    String label;
    bool match;  //For object list
    bool consider_as_object; // For blob list
    bool is_stationary;
    bool is_person;
    bool is_bag;

    int keypoints_count;
    int corner_count;
    double compactness;
    int circle_count;
    double  h_w_ratio;

    int stationary_count;



    Blobs()
    {
        match=0;
        consider_as_object=1;
        is_stationary=0;
        stationary_count=0;

        is_person=0;
        is_bag = 0;

        corner_count=0;
        compactness=0;
        circle_count=0;
        h_w_ratio=0;
    }
};

class Segmentor
{
  private:
    char  filename[256];
    IplImage *raw_image, *lab_image, *seg_image;
    CvBGCodeBookModel *model;
    vector<vector<Point> >contours;
    //vector<Vec4i> hierarchy;


  public:
    Blobs *blob_list;
    Blobs *obj_list;    //Object list
    Reference *ref;
   // Blobs *suitable_list;
    bool file_reading;
    int nframes_to_learn_bg;
    int canny_threshold;
    int frame_count;
    char fname[256],suffix[32];
    CvCapture *camera;
    Mat seg_mat,canny_op;
    Segmentor();
    ~Segmentor();

    void obtain_reference();
    void capture();
    void learn_bg_model();
    void bgfg_separate();

    void blob_mark();
    void gen_objects();
    bool bag_detector(int);

};


Segmentor::Segmentor()
{
  nframes_to_learn_bg = 149;
  raw_image=0;
  camera = 0;
  frame_count = 0;
  file_reading = false;
  canny_threshold =100;

  model =cvCreateBGCodeBookModel();
  //emperical values for our reference :-/
  model -> modMin[0] = model -> modMin[1] = model -> modMin[2] =20;
  model -> modMax[0] = model -> modMax[1] = model -> modMax[2]= 44;
  model -> cbBounds[0] = model -> cbBounds[1] = model -> cbBounds[2] =10;

 // cvNamedWindow ( "RawImage", 1 );
  cvNamedWindow ( "SegmentedCleanImage", 1 );
}


Segmentor::~Segmentor()
{
    delete[] ref;
    cvReleaseCapture( &camera );
    //cvDestroyWindow ( "RawImage");
    cvDestroyWindow ( "SegementedCleanImage");
}

void Segmentor::obtain_reference()
{
    ref = new Reference [TRAINING_DATA_SIZE];


    for(int i=1001;i<TRAINING_DATA_SIZE;i++)
    {

        char file_name[256],file_suffix[16];

        strcpy(file_name,TRAINING_PATH);
        sprintf(file_suffix,"%d",i);
        file_suffix[0]='0';
        strcat(file_name,file_suffix);
        strcat(file_name,".jpg");
        cout<<endl<<file_name;
        ref[(i%1000)].get_features(file_name);
       // cout<<"\n H/W: "<<ref[(i%1000)].h_w_ratio;
       // cout<<"\n Corners: "<<ref[(i%1000)].corner_count;
        //cout<<"\n Circles: "<<ref[(i%1000)].circle_count<<endl;

     }

}


void Segmentor::capture()
{
    if(file_reading)
    {
        //cout<<fname<<endl;
        raw_image=cvLoadImage(fname,1);
        frame_count++;
        if(raw_image == 0)
             cout<<endl<<"Image not loaded"<<endl;

    }
    else
    {
        raw_image = cvQueryFrame( camera );
        frame_count++;
    }

    \
    if(frame_count == 1 && raw_image )
    {
        lab_image = cvCloneImage (raw_image);
        seg_image = cvCreateImage( cvGetSize(raw_image), IPL_DEPTH_8U, 1);
        cvSet ( seg_image , cvScalar(255) );
    }

    //cout<<"\n before cvtcolor \n";

    //convert to lab image
    cvCvtColor( raw_image, lab_image, CV_BGR2Lab);

    //display
    //cout<<"\n converted \n";
    //cvShowImage( "RawImage" , raw_image );
    //cvWaitKey(10);
}


void Segmentor:: learn_bg_model()
{

    //cout<<"\n Learning \n";
  if(frame_count < nframes_to_learn_bg )
    cvBGCodeBookUpdate( model , lab_image);
  if(frame_count == nframes_to_learn_bg )
  {
      cout<<" >> Background Modelling Complete.... \n";
      cvBGCodeBookClearStale(model, model->t/2);
  }
    //cout<<"\n Learning complete";
}



void Segmentor::bgfg_separate()
{
  if(frame_count > nframes_to_learn_bg)
  {
    cvBGCodeBookDiff( model, lab_image, seg_image);
    cvSegmentFGMask(seg_image);
    cvShowImage( "SegmentedCleanedImage", seg_image);
    cvWaitKey(10);
  }
}


void Segmentor::blob_mark()
{
    seg_mat=Mat(seg_image);
    Mat raw_mat=Mat(raw_image);
   // people_detector(raw_mat);
   // Mat lab_mat=Mat(lab_image);
    ColorHistogram c1;

   // imshow("Lab image",lab_mat);

    findContours(seg_mat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    vector<Mat> subregions;
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    vector<vector<Point> >hull( contours.size() );

     Mat drawing = Mat::zeros( seg_mat.size(), CV_8UC3 );


    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }

    blob_list = new Blobs [contours.size()];




    for (int i = 0; i < contours.size(); i++)
    {
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
        convexHull( Mat(contours[i]), hull[i], false );
        if(frame_count%10 == 0)
        {
            // Get bounding box for contour
            //Rect roi = boundingRect(contours[i]);
            Rect roi = boundingRect(hull[i]); // This is a OpenCV function

            // Create a mask for each contour to mask out that region from image.
            Mat mask = Mat::zeros(seg_mat.size(), CV_8UC1);
            //drawContours(mask, contours, i, Scalar(255), CV_FILLED);
            drawContours(mask, hull, i, Scalar(255), CV_FILLED); // This is a OpenCV function

            // At this point, mask has value of 255 for pixels within the contour and value of 0 for those not in contour.
            // Extract region using mask for region
            Mat contourRegion;
            Mat imageROI,maskROI;
            raw_mat.copyTo(imageROI,mask);
            contourRegion = imageROI(roi);
            //maskROI = mask(roi); // Save this if you want a mask for pixels within the contour in contourRegion.

            blob_list[i].region=contourRegion.clone();
            blob_list[i].blob_id=i;
            blob_list[i].centroid=center[i];
            //blob_list[i].area = contourArea(Mat(contours[i]));
            //blob_list[i].perimeter = arcLength( contours[i], true );
            // blob_list[i].hist = c1.getHistogram(blob_list[i].region);
            blob_list[i].hist = c1.getabHistogram(blob_list[i].region);
            normalize(blob_list[i].hist,blob_list[i].hist,0,blob_list[i].hist.rows,NORM_MINMAX,-1,Mat());

            // Store contourRegion. contourRegion is a rectangular image the size of the bounding rect for the contour
            // BUT only pixels within the contour is visible. All other pixels are set to (0,0,0).
            subregions.push_back(contourRegion);

//Uncomment the following lines to view ALL regions
            /*for(int i=0;i<subregions.size();i++)
            {
                string Result;          // string which will contain the result
                ostringstream convert;   // stream used for the conversion
                convert << i;      // insert the textual representation of 'Number' in the characters in the stream
                Result = convert.str(); // set 'Result' to the contents of the stream
                //imshow(Result,subregions[i]);
                imshow(Result,blob_list[i].region);
            }*/

            blob_list[i].h_w_ratio = (double) blob_list[i].region.rows/blob_list[i].region.cols;

         /*   Mat gray;
            cvtColor(blob_list[i].region, gray, CV_BGR2GRAY);
            GaussianBlur( gray, gray, Size(9, 9), 2, 2 );

            vector<Point2f> corners_in_blob;
            goodFeaturesToTrack(gray, corners_in_blob, 5, 0.6, 3);
//Vary quality level and minDist for better results - InputArray image, OutputArray corners, int maxCorners, double qualityLevel, double minDistance, InputArray mask=noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04
            blob_list[i].corner_count = corners_in_blob.size();

            vector<Vec3f> circles;
            HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, gray.rows/4, 200, 100 );
            blob_list[i].circle_count = circles.size();
*/




        }
    }

    namedWindow( "Contours",1 );
    add(drawing,Mat(raw_image),drawing);
    imshow( "Contours", drawing );
    //waitKey(10);

}



void Segmentor::gen_objects()
{
   // suitable_list = new Blobs [contours.size()];

    double hist_dist=0.0,min_hist_dist=0.0,centroid_dist=0.0,min_centroid_dist = 0.0;
    int pos,count=0,i,j;

    if(frame_count == nframes_to_learn_bg+1)
    {
            obj_list=blob_list;
            return;
    }

    for ( i = 0; i < contours.size()  ; i++)
    {

        for(j = 0; j<contours.size();j++)
        {
            centroid_dist = sqrt((pow((obj_list[j].centroid.x -  blob_list[i].centroid.x),2)+(pow((obj_list[j].centroid.y - blob_list[i].centroid.y),2))));
            //cout<<"\n Blob "<<i<<" , Object "<<j<<" ---------> "<<centroid_dist;
            //cout<<"\n Obj X: "<<obj_list[j].centroid.x<<"Obj Y: "<<obj_list[j].centroid.y<<"\t Blob X: "<<blob_list[i].centroid.x<<" Blob Y:"<<blob_list[i].centroid.y;

            // hist_dist = compareHist(blob_list[i].hist,obj_list[j].hist,CV_COMP_INTERSECT);

            if(centroid_dist <= 5.0)
            {
                cout<<"\n\n\n Stationary Count: "<<obj_list[j].stationary_count;
                blob_list[i].is_stationary = 1;
                blob_list[i].stationary_count = obj_list[j].stationary_count+1;
                if(blob_list[i].stationary_count > STATIONARY_THRESHOLD )
                {
                      count++;
                    blob_list[i].consider_as_object=1;
                    send_msg(STATIONARY);
                    blob_list[i].is_person = people_detector(blob_list[i].region);
                    if(blob_list[i].is_person)
                    {
                         send_msg(PERSON);
                         //add putText here
                    }
                    else
                    {
                        blob_list[i].is_bag = bag_detector(i);
                        if(blob_list[i].is_bag)
                            send_msg(BAG);
                        //Add putText here
                     }
                }
            }
            else
            {
                blob_list[i].stationary_count = 0;
                blob_list[i].is_stationary =0;
                blob_list[i].consider_as_object=0;
            }

        }

        //blob_list[i].label = obj_list[pos].label;
    }


    delete[] obj_list;
    obj_list= blob_list;
    cout<<endl<<endl;

}


int main( int argc, char *argv[])
{


    if(send_msg(TEST)==-1)
    {
        cout<<"\n E: Message Board not found!!!! \n Please start the Message Board Process first"<<endl<<endl;
        exit(-1);
    }





    Segmentor s1;
    //check for successful camera open
    if( argc == 1 )
    {


        s1.camera = cvCaptureFromCAM( 0 );
        if ( !s1.camera )
        {
          cout << " E: Camera capture failed!!!!"<<endl<<endl;
          exit (-1);
        }

        printf("\n >> Obtaining Training Data.... \n");
        s1.obtain_reference();
        printf("\n >> Modelling Background.... \n");
        while(1)
        {

            s1.capture();

            if(s1.frame_count <= s1.nframes_to_learn_bg )
            {
                printf("---------------- %d \n",s1.frame_count);
                s1.learn_bg_model();

            }
            else
            {
                s1.bgfg_separate();
                s1.blob_mark();
                if(s1.frame_count % 10 == 0)
                    s1.gen_objects();

            }
        }
    }



  else
  {
    s1.file_reading = true;

    for(int i=10000;i<=13020;i++)
    {
        strcpy(s1.fname,DATASET_PATH);
        sprintf(s1.suffix,"%d",i);
        s1.suffix[0]='0';
        strcat(s1.fname,s1.suffix);
        strcat(s1.fname,".jpeg");
        cout<<endl<<s1.fname<<endl;

        s1.capture();
        //cout<<"\n Capture done.... ";

        s1.learn_bg_model();
        s1.bgfg_separate();
        s1.blob_mark();
    }
  }
  return 0;
}


int send_msg(int index)
{
    int create_socket;
    char to_be_sent[256];
    struct sockaddr_in address;

    if ((create_socket = socket(AF_INET,SOCK_STREAM,0)) < 0)
    {
        cout<<"\n E: Socket creation failed!!!! ";
    }

    cout<<"\n > Socket Created .... ";

    address.sin_family = AF_INET;
    address.sin_port = htons(15000);
    inet_pton(AF_INET,LOCALHOST,&address.sin_addr); //Converts the network address from dotted decimal to binary

    if (connect(create_socket,(struct sockaddr *) &address,sizeof(address)) < 0)
        return -1;


    cout<<"\n > Connection with Message Board ("<<LOCALHOST<<") accepted....";


    sprintf(to_be_sent,"%d",index);
    if(send(create_socket, to_be_sent, sizeof(to_be_sent), 0)==0)
    {
        cout<<"\n ~ Failed to send message!!!! ";
        exit(-1);
    }
    sleep(1);
    cout<<"\n > Message sent...";
    cout<<endl<<endl;

    return close(create_socket);

}

bool Segmentor::bag_detector(int index)
{
    double hist_dist;

    //Mat img,blob;
    //blob_list[index].hist.convertTo(blob,CV_32F);
    bool result =0;

    for(int i=1001;i<TRAINING_DATA_SIZE-1;i++)
    {
        //ref[i].hist.convertTo(img,CV_32F);
        //hist_dist = compareHist(img,blob,CV_COMP_CORREL);
        hist_dist = compareHist(ref[(i%1000)].hist, blob_list[index].hist,CV_COMP_CORREL);
        cout<<"\n Hist Dist "<<(i%1000)<<": "<<hist_dist;
        if(hist_dist > 0.65)
        {
            result=1;
            break;
        }
    }

    return result;


}

bool people_detector (Mat frame)
{

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

        vector<Rect> found, found_filtered;
        hog.detectMultiScale(frame, found, 0, Size(8,8), Size(32,32), 1.05, 2);

        size_t i, j;
        for (i=0; i<found.size(); i++)
        {
            Rect r = found[i];
            for (j=0; j<found.size(); j++)
                if (j!=i && (r & found[j])==r)
                    break;
            if (j==found.size())
                found_filtered.push_back(r);
        }
        for (i=0; i<found_filtered.size(); i++)
        {
        Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.06);
        r.height = cvRound(r.height*0.9);
        rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
    }
        imshow("Current ", frame);

        if(found_filtered.size()!=0)
            return 1;

        return 0;

}

