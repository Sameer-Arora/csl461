#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

void print_ma(Mat src){
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			cout<<src.at<float>(i,j)<<" ";
		}
		cout<<endl;
	}
}

uchar bilinear_util(int p1,int p2,double alp,double bet,Mat sr){           
		return round( (1-alp)*(1-bet)*sr.at<uchar>(p1+1,p2+1) + (alp)*(1-bet)*sr.at<uchar>(p1+2,p2+1)
			+(1-alp)*(bet)*sr.at<uchar>(p1+1,p2+2)+(alp)*(bet)*sr.at<uchar>(p1+2,p2+2) );
}

uchar bilinear(double i1,double j1,Mat sr){
	int rows=sr.rows-2;
	int cols=sr.cols-2;

	int p1; double alp;                        // p1,p2 is start or top-left of four nieghbours.

	if(i1<-0.5){                     //boundary condition.
		p1=-1;  alp=1.5 +i1;
	}
	else if(i1 +1.5 > rows ){
		p1=rows -1;  alp=i1+1.5-rows;
	}
	else if( (i1+0.5) -int(i1+0.5)==0 ){
		p1=(i1<0)? 0:int(i1)+1, alp=0;
	}
	else if( int(i1+0.5) >int(i1) ){
		 p1=(i1<0)? 0:int(i1)+1;
		 alp=i1-(p1-0.5);
	}
	else{
		p1=(i1<0)? 0:int(i1);
		alp=i1-(p1-0.5);
	}

	int p2; double bet;
	if(j1<-0.5){  //boundary condition.
		p2=-1;  bet=1.5 +j1;
	}else if(j1 +1.5 > cols){
		p2=cols -1; bet=i1+1.5-cols;
	}else if( (j1+0.5) -int(j1+0.5)==0 ){
		p2=(j1<0)? 0:int(j1)+1,bet=0;
	}else if( int(j1+0.5) >int(j1) ){
		 p2=(j1<0)? 0:int(j1)+1;
		 bet=j1-(p2-0.5);
	}else{
		p2=(j1<0)? 0:int(j1);
		 bet=j1-(p2-0.5);
	}
	return bilinear_util(p1,p2,alp,bet,sr);
}


Mat create(Mat src,Mat tran,int& x_m,int& y_m ){
	int row,col;
	int x_mx,x_min;
	float a,b,c;
	
	a=tran.at<float>(0,0);
	b=tran.at<float>(0,1);
	c=tran.at<float>(0,2);

	cout<<a<<" "<<b<<" "<<c<<endl;
	
	if( a*b < 0){
		if(a>0){
			x_mx= a*src.rows;
			x_min=b*src.cols;
		}else{
			x_mx=b*src.cols;
			x_min=a*src.rows;
		}
	}
	else if(a*b==0){
		if(a==0){
			x_mx=b*src.cols;
			x_min=0;
		}
		else{
			x_mx=a*src.rows;
			x_min=0;
		}
	}
	else{
		if(a>0){
			x_mx= a*src.rows + b*src.cols;
			x_min=0;
		}else{
			x_mx=0;
			x_min=a*src.rows + b*src.cols;
		}
	}
	row=round(x_mx-x_min+c);
	x_m=x_min;

	a=tran.at<float>(1,0);
	b=tran.at<float>(1,1);
	c=tran.at<float>(1,2);
	cout<<a<<" "<<b<<" "<<c<<endl;

	if( a*b < 0){
		if(a>0){
			x_mx= a*src.rows;
			x_min=b*src.cols;
		}else{
			x_mx=b*src.cols;
			x_min=a*src.rows;
		}
	}
	else if(a*b==0){
		if(a==0){
			x_mx=b*src.cols;
			x_min=0;
		}
		else{
			x_mx=a*src.rows;
			x_min=0;
		}
	}
	else{
		if(a>0){
			x_mx= a*src.rows + b*src.cols;
			x_min=0;
		}else{
			x_mx=0;
			x_min=a*src.rows + b*src.cols;
		}
	}
	col=round(x_mx-x_min+c);
	y_m=x_min;

	cout<<src.rows<<" "<<src.cols<<" "<<row<<" "<<col<<"\n";
	Mat dst=Mat::zeros(row,col,CV_8U);
	return dst;

}


  // x,y's are from the original image whereas x_,y_'s are from distorted ones.
Mat tie_points(float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4,
	int x1_,int y1_,int x2_,int y2_,int x3_,int y3_,int x4_,int y4_){  //returns transformation matrix.

	Mat t=Mat::zeros(8,1,CV_32F);
	t.at<float>(0,0)=x1_;
	t.at<float>(1,0)=y1_;
	t.at<float>(2,0)=x2_;
	t.at<float>(3,0)=y2_;
	t.at<float>(4,0)=x3_;
	t.at<float>(5,0)=y3_;
	t.at<float>(6,0)=x4_;
	t.at<float>(7,0)=y4_;
	
	Mat tran=Mat::zeros(8,8,CV_32F);
	tran.at<float>(0,0)=x1;
	tran.at<float>(0,1)=y1;
	tran.at<float>(0,2)=x1*y1;
	tran.at<float>(0,3)=1;
	tran.at<float>(1,4)=x1;
	tran.at<float>(1,5)=y1;
	tran.at<float>(1,6)=x1*y1;
	tran.at<float>(1,7)=1;
	tran.at<float>(2,0)=x2;
	tran.at<float>(2,1)=y2;
	tran.at<float>(2,2)=x2*y2;
	tran.at<float>(2,3)=1;
	tran.at<float>(3,4)=x2;
	tran.at<float>(3,5)=y2;
	tran.at<float>(3,6)=x2*y2;
	tran.at<float>(3,7)=1;
	tran.at<float>(4,0)=x3;
	tran.at<float>(4,1)=y3;
	tran.at<float>(4,2)=x3*y3;
	tran.at<float>(4,3)=1;
	tran.at<float>(5,4)=x3;
	tran.at<float>(5,5)=y3;
	tran.at<float>(5,6)=x3*y3;
	tran.at<float>(5,7)=1;
	tran.at<float>(6,0)=x4;
	tran.at<float>(6,1)=y4;
	tran.at<float>(6,2)=x4*y4;
	tran.at<float>(6,3)=1;
	tran.at<float>(7,4)=x4;
	tran.at<float>(7,5)=y4;
	tran.at<float>(7,6)=x4*y4;

	Mat con= tran.inv() * t;

	return con;
}

void get_points(int& x1, int& y1,int& x2, int& y2,int& x3, int& y3,Vec6f t ){
   
        x1 =cvRound(t[0]);
        y1=cvRound(t[1]);
        x2 =cvRound(t[2]);
        y2=cvRound(t[3]);
        x3 =cvRound(t[4]);
        y3=cvRound(t[5]);
}     



double dist(float x1,float y1,float x2,float y2){
	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

double max(double d1,double d2,double d3){
	double max=d1;
	if(d2>max) max=d2;
	if(d3>max) max=d3;
	return max;
}

double min(double d1,double d2,double d3){
	double min=d1;
	if(d2 < min) min=d2;
	if(d3 <min ) min=d3;
	return min;
}

double area(double x1,double y1,double x2,double y2,double x3,double y3){
	return abs( x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2) )/2 ;
}

bool inside(int x,int y,double x1,double y1,double x2,double y2,double x3,double y3){
	return ( area(x-0.5,y-0.5,x1,y1,x2,y2) + area(x-0.5,y-0.5,x3,y3,x2,y2) + area(x-0.5,y-0.5,x1,y1,x3,y3) == area(x3,y3,x1,y1,x2,y2) ) ;  
}

Mat frame(Mat src,Mat dst,double alp, Point** mapping, vector<Vec6f> triangleList ){

	Mat fra=Mat::zeros(src.rows,src.cols,CV_8U);

	int x1,y1,x2,y2,x3,y3,x4,y4,x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_;
	double i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,i_x4,i_y4;

    for( size_t i = 0; i < triangleList.size(); i++ ){
    	Vec6f trian=triangleList[i];
    	get_points(x1,y1,x2,y2,x3,y3,trian);

    	x4=(x1+x2+x3)/3;
		y4=(y1+y2+y3)/3;
		
    	x1_=mapping[x1][y1].x;
    	y1_=mapping[x1][y1].y;

    	x2_=mapping[x2][y2].x;
    	y2_=mapping[x2][y2].y;

    	x3_=mapping[x3][y3].x;
    	y3_=mapping[x3][y3].y;

		// choosing the 4th piont as centroid.
		x4_=(x1_+x2_+x3_)/3;
		y4_=(y1_+y2_+y3_)/3;

    	i_x1= x1_*alp +x1*(1-alp);
		i_y1= y1_*alp +y1*(1-alp);

		i_x2= x2_*alp +x2*(1-alp);
		i_y2= y2_*alp +y2*(1-alp);

		i_x3= x3_*alp +x3*(1-alp);
		i_y3= y3_*alp +y3*(1-alp);

		i_x4= x4_*alp +x4*(1-alp);	
		i_y4= y4_*alp +y4*(1-alp);

		cout<<"imA "<<x1<<","<<y1<<" "<<x2<<","<<y2<<" "<<x3<<","<<y3<<" "<<x4<<","<<y4<<" \n";
		cout<<"imB "<<x1_<<","<<y1_<<" "<<x2_<<","<<y2_<<" "<<x3_<<","<<y3_<<" "<<x4_<<","<<y4_<<" \n";
		cout<<"inter "<<i_x1<<","<<i_y1<<" "<<i_x2<<","<<i_y2<<" "<<i_x3<<","<<i_y3<<" "<<i_x4<<","<<i_y4<<" \n";

		Mat inv_A=tie_points(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,i_x4,i_y4,x1,y1,x2,y2,x3,y3,x4,y4);

		Mat inv_B=tie_points(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,i_x4,i_y4,x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_);

		int x_max= round( max(i_x1,i_x2,i_x3) );
		int x_min= round( min(i_x1,i_x2,i_x3) );

		int y_max= round( max(i_y1,i_y2,i_y3) );
		int y_min= round( min(i_y1,i_y2,i_y3) );

		for(int i=x_min;i<=x_max;i++){
			for(int j=y_min;j<=y_max;j++){
				if(inside(i,j,i_x1,i_y1,i_x2,i_y2,i_x3,i_y3)){
					double i1= inv_A.at<float>(0,0)* i +inv_A.at<float>(1,0)* j+inv_A.at<float>(2,0)* i*j+inv_A.at<float>(3,0); 			
					double j1= inv_A.at<float>(4,0)* i +inv_A.at<float>(5,0)* j+inv_A.at<float>(6,0)* i*j+inv_A.at<float>(7,0); 			
			 		fra.at<uchar>(i,j)= bilinear(i1,j1,src);      //finding the values for original image from intensities of distorted ones.
			 		
			 		i1= inv_B.at<float>(0,0)* i +inv_B.at<float>(1,0)* j+inv_B.at<float>(2,0)* i*j+inv_B.at<float>(3,0); 			
					j1= inv_B.at<float>(4,0)* i +inv_B.at<float>(5,0)* j+inv_B.at<float>(6,0)* i*j+inv_B.at<float>(7,0); 			
			 		fra.at<uchar>(i,j)= ( bilinear(i1,j1,dst)*alp +fra.at<uchar>(i,j) *(1-alp) );      //finding the values for original image from intensities of distorted ones.
			 			
				}
			}
		}


    }
	
 	return fra;

}

float normalize(float& x,int limit){
	if(x<0) x=0;
	if(x>=limit) x=limit-1;
}

// Get delaunay triangles
static vector<Vec6f> get_delaunay( Mat& img, Subdiv2D& subdiv )
{
 
    vector<Vec6f> triangleList;
    vector<Vec6f> resList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
 	
 	Scalar color(255,255,255);


    for( size_t i = 0; i < triangleList.size(); i++ ){

    	Vec6f t = triangleList[i];
       /* normalize(t[0],size.width);
        normalize(t[1],size.height);
        normalize(t[2],size.width);
        normalize(t[3],size.height);
        normalize(t[4],size.width);
        normalize(t[5],size.height);
       */ 
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        
         

        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {	
        	line(img, pt[0], pt[1],color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], color , 1, CV_AA, 0);
            cout<<"tringle"<<i<<endl;
        
            cout<<pt[0].x<<" "<<pt[0].y<<endl;
	        cout<<pt[1].x<<" "<<pt[1].y<<endl;
	        cout<<pt[2].x<<" "<<pt[2].y<<endl;
            resList.push_back(t);
        }
    }
 	return resList;

}

int main(int argc, char** argv){
	ifstream fin;
	fin.open("points_A.txt");

	ifstream fin1;
	fin1.open("points_B.txt");

 	Mat imageA,image=imread("bush.jpg");
 	if (image.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	cvtColor( image, imageA, CV_BGR2GRAY );


 	Mat imageB=imread("bush.jpg");
 	if (imageB.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	cvtColor( imageB, imageB, CV_BGR2GRAY );

	String win_imageA = "imageA"; 
	namedWindow(win_imageA); 
	
	String win_imageB = " imageB "; 
	namedWindow(win_imageB); 

	String win_intermediate = "intermediate"; 
	namedWindow(win_intermediate); 
/*
	String windowName2 = "opencv g_imaged error  "; 
	namedWindow(windowName2); 

	String windowName = "opencv g_imaged trans  "; 
	namedWindow(windowName); */
	
	Size dsize(0,0);
	
 	Mat res,fra;
 	int num;

    // Rectangle to be used with Subdiv2D
    Size size = imageA.size();
    
    Rect rect(0, 0, size.width, size.height);

 	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    
 	Point** mapping=new Point*[imageA.rows];
	for(int i = 0; i < imageA.rows; ++i)
    	mapping[i] = new Point[imageA.cols];

 	std::vector<Point> points_A;

 	int x,y,x_,y_;
 	std::string line;

	while(!fin.eof() && !fin1.eof() ){
		getline(fin,line);
		istringstream is( line );
		while( is >> x >> y >> x_ >> y_) {
		//  cout<< x <<" "<< y <<" "<< x_<<" " << y_<<endl;
		  points_A.push_back(Point(x,y));
		  Point p1=Point(x_,y_);
		  mapping[x][y]=p1;
		}
	 	fin>>num;

        Mat g_imagecopy = imageA.clone();

		 // Insert points into subdiv
	    for( vector<Point>::iterator it = points_A.begin(); it != points_A.end(); it++)
	    {
	        subdiv.insert(*it);
        }
        vector<Vec6f> triangleList=get_delaunay( g_imagecopy , subdiv );  // Get delaunay triangles
        imshow(win_imageA, g_imagecopy );
        waitKey(0);
    
	         
	    
		double alp=0;
	 	for(int i=0;i<=num;i++){
	 		alp+=1.0/(num+1);
	 		fra=frame(imageA,imageB,alp,mapping,triangleList);
	 		
	 		//cout<<"rows-cols"<<fra.rows<<" "<<fra.cols<<" "<<g_image.rows<<" "<<g_image.cols<<"\n";
	 	 	
	 	 	imshow(win_imageA,imageA );
	 	 	imshow(win_imageB,imageB );
	 	 	imshow(win_intermediate,fra );
	 	 	waitKey(0);
	 		}
	 
	 }
					

}

