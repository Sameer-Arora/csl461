#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;


// Draw a single point
static void draw_point( Mat& img, Point fp )
{	
	 Scalar color(255,255,255);
    circle( img, fp, 4, color, CV_FILLED, CV_AA, 0 );
}
 
void print_ma(Mat src){
	for(int i=0;i<src.rows;i++){
		for(int j=0;j<src.cols;j++){
			cout<<src.at<double>(i,j)<<" ";
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

/*int max(int d1,int d2,int d3){
	int max=d1;
	if(d2>max) max=d2;
	if(d3>max) max=d3;
	return max;
}

int min(int d1,int d2,int d3){
	int min=d1;
	if(d2 < min) min=d2;
	if(d3 <min ) min=d3;
	return min;
}
*/
double area(int x1,int y1,int x2,int y2,int x3,int y3){
	return abs( x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2) )/2.0 ;
}

bool inside(int x,int y,int x1,int y1,double x2,int y2,int x3,int y3){
	return ( area(x,y,x1,y1,x2,y2) + area(x,y,x3,y3,x2,y2) + area(x,y,x1,y1,x3,y3) - area(x3,y3,x1,y1,x2,y2) < 0.0001 ) ;  
}

vector<Vec6f> get_intermediate_triangulation( Mat img,Mat img2,vector<Point> points_A,Point** mapping,Point** inv_mapping,Subdiv2D& subdiv,double alp){
	
	int x1,y1,x1_,y1_;
	int i_x1,i_y1;

 	for( vector<Point>::iterator it = points_A.begin(); it != points_A.end(); it++){
	        x1=(*it).x;
	        y1=(*it).y;
	        x1_=mapping[x1][y1].x; y1_=mapping[x1][y1].y;

	        i_x1= x1_*alp +x1*(1-alp);
			i_y1= y1_*alp +y1*(1-alp);
/*
				cout<<"imA "<<y1<<","<<x1<<" \n";
				cout<<"imB "<<y1_<<","<<x1_<<" \n";
				cout<<"inter "<<i_y1<<","<<i_x1<<" \n";

*/			inv_mapping[i_x1][i_y1]=Point(x1,y1);
			subdiv.insert(Point(i_x1,i_y1));

        }

    vector<Vec6f> triangleList;
    vector<Vec6f> resList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
 	
 	Scalar color(255,255,255);


    for( size_t i = 0; i < triangleList.size(); i++ ){

    	Vec6f t = triangleList[i];

        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]) )
        {	
        	line(img, pt[0], pt[1],color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], color , 1, CV_AA, 0);
        	line(img2, pt[0], pt[1],color, 1, CV_AA, 0);
            line(img2, pt[1], pt[2], color, 1, CV_AA, 0);
            line(img2, pt[2], pt[0], color , 1, CV_AA, 0);
            resList.push_back(t);
        }
    }
 	return resList;

}

Mat trans_mat(float x1,float y1,float x2,float y2,float x3,float y3,float x1_,float y1_,float x2_,float y2_,float x3_,float y3_){  //returns transformation matrix.
	Mat res=Mat::eye(3,3,CV_32F);

	Mat t=Mat::zeros(6,1,CV_32F);
	t.at<float>(0,0)=x1_;
	t.at<float>(1,0)=y1_;
	t.at<float>(2,0)=x2_;
	t.at<float>(3,0)=y2_;
	t.at<float>(4,0)=x3_;
	t.at<float>(5,0)=y3_;

	Mat tran=Mat::zeros(6,6,CV_32F);
	tran.at<float>(0,0)=x1;
	tran.at<float>(0,1)=y1;
	tran.at<float>(0,2)=1;
	tran.at<float>(1,3)=x1;
	tran.at<float>(1,4)=y1;
	tran.at<float>(1,5)=1;
	tran.at<float>(2,0)=x2;
	tran.at<float>(2,1)=y2;
	tran.at<float>(2,2)=1;
	tran.at<float>(3,3)=x2;
	tran.at<float>(3,4)=y2;
	tran.at<float>(3,5)=1;	
	tran.at<float>(4,0)=x3;
	tran.at<float>(4,1)=y3;
	tran.at<float>(4,2)=1;
	tran.at<float>(5,3)=x3;
	tran.at<float>(5,4)=y3;
	tran.at<float>(5,5)=1;

	Mat con= tran.inv()*t;
/*
 	for(int i=0;i<2;i++){
 		for(int j=0;j<3;j++){
			res.at<float>(i,j)=con.at<float>(3*i+j);
			cout<<res.at<float>(i,j)<<" ";
 			}
 			cout<<endl;
 		}*/
	return con;
}


Mat frame(Mat src,Mat dst,double alp, Point** mapping, vector<Point> points_A ){

	ofstream fout("out.txt");

	Mat fra=Mat::zeros(src.rows,src.cols,CV_8U);
	Mat a_inv=Mat::zeros(src.rows,src.cols,CV_8U);
	Mat b_inv=Mat::zeros(src.rows,src.cols,CV_8U);

    String win_imageA_in = "imageA_inv"; 
	namedWindow(win_imageA_in); 
	
    String win_imageB_in = "imagB_inv"; 
	namedWindow(win_imageB_in); 

    String win_imagefr = "imagfr"; 
	namedWindow(win_imagefr); 
	
	// Rectangle to be used with Subdiv2D
    Size size = src.size();
    
    Rect rect(0, 0, size.width, size.height);

 	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    	

 	Point** inv_mapping=new Point*[src.cols];
	for(int i = 0; i < src.cols; ++i)
	   	inv_mapping[i] = new Point[src.rows];

	Mat src_clo=src.clone();   
	Mat dst_clo=dst.clone();   

	vector<Vec6f> triangleList=get_intermediate_triangulation(src_clo,dst_clo,points_A,mapping,inv_mapping,subdiv,alp);
	/*imshow(win_imageA_in,src_clo);
	imshow(win_imageB_in,dst_clo);
	waitKey(0);
*/
	int x1,y1,x2,y2,x3,y3,x4,y4,x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_;
	int i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,i_x4,i_y4;

    for( size_t i = 0; i < triangleList.size(); i++ ){
    	Vec6f trian=triangleList[i];
    	get_points(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,trian);
	
		x1=inv_mapping[i_x1][i_y1].x;
		y1=inv_mapping[i_x1][i_y1].y;


		x2=inv_mapping[i_x2][i_y2].x;
		y2=inv_mapping[i_x2][i_y2].y;

		x3=inv_mapping[i_x3][i_y3].x;
		y3=inv_mapping[i_x3][i_y3].y;
	
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
/*
    	i_x4= x4_*alp +x4*(1-alp);	
		i_y4= y4_*alp +y4*(1-alp);*/
    	
    	i_x4=(i_x1+i_x2+i_x3)/3;
    	i_y4=(i_y1+i_y2+i_y3)/3;
		
		Point fp(i_x1,i_y1);
	
	//waitKey(0);


		Mat inv_A=trans_mat(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,x1,y1,x2,y2,x3,y3);
		/*std::vector<Point2f> s;
		std::vector<Point2f> d;
		std::vector<Point2f> in;
		s.push_back(Point2f(x1,y1));s.push_back(Point2f(x2,y2));s.push_back(Point2f(x3,y3));
		d.push_back(Point2f(y1_,x1_));d.push_back(Point2f(y2_,x2_));d.push_back(Point2f(y3_,x3_));
		in.push_back(Point2f(i_x1,i_y1));in.push_back(Point2f(i_x2,i_y2));in.push_back(Point2f(i_x3,i_y3));
		std::vector<Point2f>::iterator it,it2;
		for(it=s.begin(),it2=in.begin();it2!=in.end() && it!=s.end();it++,it2++){
			cout<<it->x<<" "<<it->y<<endl;
			cout<<it2->x<<" "<<it2->y<<endl;
		} 
/*		for(it=d.begin();it!=d.end();it++){
			cout<<it->x<<" "<<it->y<<endl;
		} 
	
		Mat inv_A =getAffineTransform(s,in);
		print_ma(inv_A);		
*/
		Mat inv_B=trans_mat(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,x1_,y1_,x2_,y2_,x3_,y3_);

		int x_max= max(i_x1,max(i_x2,i_x3)) ;
		int x_min= min(i_x1,min(i_x2,i_x3)) ;

		int y_max= max(i_y1,max(i_y2,i_y3)) ;
		int y_min= min(i_y1,min(i_y2,i_y3)) ;

		int x11=(x4_+x1_)/2;
		int y11=(y4_+y1_)/2;

		cout<<"imB "<<x11<<" "<<y11<<" "<<x4_<<" "<<y4_<<" "<<x1_<<","<<y1_<<" "<<x2_<<","<<y2_<<" "<<x3_<<","<<y3_<<" \n";

		x1_= inv_B.at<float>(0,0)* i_x1 +inv_B.at<float>(1,0)* i_y1 +inv_B.at<float>(2,0); 			
		y1_= inv_B.at<float>(3,0)* i_x1 +inv_B.at<float>(4,0)* i_y1+inv_B.at<float>(5,0); 			
		
		x2_= inv_B.at<float>(0,0)* i_x2 +inv_B.at<float>(1,0)* i_y2 +inv_B.at<float>(2,0); 			
		y2_= inv_B.at<float>(3,0)* i_x2 +inv_B.at<float>(4,0)* i_y2+inv_B.at<float>(5,0); 			
		
		x3_= inv_B.at<float>(0,0)* i_x3 +inv_B.at<float>(1,0)* i_y3 +inv_B.at<float>(2,0); 			
		y3_= inv_B.at<float>(3,0)* i_x3 +inv_B.at<float>(4,0)* i_y3+inv_B.at<float>(5,0); 			
		
		x4_=(x1_+x2_+x3_)/3;
		y4_=(y1_+y2_+y3_)/3;
		//cout<<"imA "<<x1<<","<<y1<<" "<<x2<<","<<y2<<" "<<x3<<","<<y3<<" \n";

		i_x4=(i_x1+i_x4)/2;
		i_y4=(i_y1+i_y4)/2;

		x11= inv_B.at<float>(0,0)* i_x4 +inv_B.at<float>(1,0)* i_y4 +inv_B.at<float>(2,0); 			
		y11= inv_B.at<float>(3,0)* i_x4 +inv_B.at<float>(4,0)* i_y4+inv_B.at<float>(5,0); 			
		
		cout<<"imB "<<x11<<" "<<y11<<" "<<x4_<<" "<<y4_<<" "<<x1_<<","<<y1_<<" "<<x2_<<","<<y2_<<" "<<x3_<<","<<y3_<<" \n";
		cout<<"inter "<<i_x4<<" "<<i_y4<<" "<<i_x1<<","<<i_y1<<" "<<i_x2<<","<<i_y2<<" "<<i_x3<<","<<i_y3<<" \n";

		
		//cout<<i_x1<<" "<<i_y1<<" "<<i_x2<<" "<<i_y2<<" "<<i_x3<<" "<<i_y3<<" "<<endl;

		cout<<"sqr "<<x_min<<" "<<x_max<< "    "<<y_min<<" "<<y_max<<endl; 

		for(int i=y_min;i<=y_max;i++){
			for(int j=x_min;j<=x_max;j++){
				if(inside(j,i,i_x1,i_y1,i_x2,i_y2,i_x3,i_y3)){
					/*fra.at<uchar>(i,j)=0;
					a_inv.at<uchar>(i,j)=0;
					b_inv.at<uchar>(i,j)=0;
					Mat m=Mat::ones(3,1,CV_64F);
					m.at<double>(0,0)=i;
					m.at<double>(1,0)=j;

					Mat n= inv_A * m;
					double i1=n.at<double>(0,0);
					double j1=n.at<double>(1,0);
					
					*/
					double j1= inv_A.at<float>(0,0)* j +inv_A.at<float>(1,0)* i+inv_A.at<float>(2,0); 			
					double i1= inv_A.at<float>(3,0)* j +inv_A.at<float>(4,0)* i+inv_A.at<float>(5,0); 			
			 		//draw_point(a_inv,Point(round(i1),round(j1) ) );
			 		a_inv.at<uchar>(i,j)=bilinear(i1,j1,src);      //finding the values for original image from intensities of distorted ones.
			 		
			 		//cout<<i<<" "<<j<<" -- "<<i1<<" "<<j1<<endl;

			 		j1= inv_B.at<float>(0,0)* j +inv_B.at<float>(1,0)* i+inv_B.at<float>(2,0); 			
					i1= inv_B.at<float>(3,0)* j +inv_B.at<float>(4,0)* i+inv_B.at<float>(5,0); 			
			 		b_inv.at<uchar>(i,j)= bilinear(i1,j1,dst) ;      //finding the values for original image from intensities of distorted ones.
			 		fra.at<uchar>(i,j)= a_inv.at<uchar>(i,j)*(1-alp)+b_inv.at<uchar>(i,j)*alp ;      //finding the values for original image from intensities of distorted ones.
				}
			}
		}
		
		//fout<<endl;

		fp=Point(x1,y1);
		draw_point( a_inv, fp );
		fp= Point(x2,y2);
		draw_point( a_inv,fp );
		fp= Point(x3,y3);
		draw_point( a_inv,fp );

		fp=Point(x1_,y1_);
		draw_point( b_inv, fp );
		fp= Point(x2_,y2_);
		draw_point( b_inv,fp );
		fp= Point(x3_,y3_);
		draw_point( b_inv,fp );
		imshow(win_imagefr,fra );
		/*
	imshow(win_imageA_in,a_inv );
	imshow(win_imageB_in,b_inv );
	waitKey(0);	
*/
    }
	imshow(win_imageA_in,a_inv );
	imshow(win_imageB_in,b_inv );
	waitKey(100);	
 	return fra;

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
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
        
         

        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {	
        	line(img, pt[0], pt[1],color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], color , 1, CV_AA, 0);
            /*cout<<"tringle"<<i<<endl;
        
            cout<<pt[0].x<<" "<<pt[0].y<<endl;
	        cout<<pt[1].x<<" "<<pt[1].y<<endl;
	        cout<<pt[2].x<<" "<<pt[2].y<<endl;
            */resList.push_back(t);
        }
                
        /*if(rect.contains(Point(0,0))) cout<<"fsafsda\n";
        if(rect.contains(Point(size.width-1,size.height-1) )) cout<<"sda\n";
*/
    }
 	return resList;

}

int main(int argc, char** argv){
	ifstream fin;
	fin.open("hillary_clinton.jpg.txt");

	ifstream fin1;
	fin1.open("ted_cruz.jpg.txt");

 	
 	Mat imageA,image=imread("hillary_clinton.jpg");
 	if (image.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	cvtColor( image, imageA, CV_BGR2GRAY );


 	Mat imageB=imread("ted_cruz.jpg");
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
 	int num=10;

    // Rectangle to be used with Subdiv2D
    Size size = image.size();
    
    Rect rect(0, 0, size.width, size.height);

 	// Create an instance of Subdiv2D
    Subdiv2D subdiv_A(rect);
    Subdiv2D subdiv_B(rect);
    
    cout<<image.rows<<" "<<image.cols<<endl;

 	Point** mapping=new Point*[image.cols];
	for(int i = 0; i < image.cols; ++i)
    	mapping[i] = new Point[image.rows];

 	std::vector<Point> points_A;
 	std::vector<Point> points_B;

 	int x,y,x_,y_,x_max=0,y_max=0;
 	std::string line;

	while(!fin.eof() && !fin1.eof() ){
		fin>>y>>x;
		fin1>>y_>>x_;
		//cout<< x <<" "<< y <<" "<< x_<<" " << y_<<endl;
	    points_A.push_back(Point(y,x));
	    //draw_point(imageA,Point(x,y));
	    points_B.push_back(Point(y_,x_));
	    //draw_point(imageB,Point(x_,y_));
    	Point p1=Point(y_,x_);
    	mapping[y][x]=p1;
    	if(x_max < x ) x_max=x;
    	if(y_max < y ) y_max=y;
	}	
	cout<<x_max<<" "<<y_max<<endl;
        Mat g_imagecopy = imageA.clone();

		 // Insert points into subdiv
	    for( vector<Point>::iterator it = points_A.begin(); it != points_A.end(); it++)
	    {
	        subdiv_A.insert(*it);
        }
        vector<Vec6f> triangleList=get_delaunay( g_imagecopy , subdiv_A );  // Get delaunay triangles
        imshow(win_imageA, g_imagecopy );
        cout<<triangleList.size()<<endl;
    
       /* g_imagecopy = imageB.clone();

		 // Insert points into subdiv
	    for( vector<Point>::iterator it = points_B.begin(); it != points_B.end(); it++)
	    {
	        subdiv_B.insert(*it);
        }
        get_delaunay( g_imagecopy , subdiv_B );
        imshow(win_imageB, g_imagecopy );
       */ waitKey(0);
	         
	    double alp=0;
	 	for(int i=0;i<=num+1;i++){
	 		fra=frame(imageA,imageB,alp,mapping,points_A);
	 		alp+=1.0/(num+1);
	 		
	 		//cout<<"rows-cols"<<fra.rows<<" "<<fra.cols<<" "<<g_image.rows<<" "<<g_image.cols<<"\n";
	 	 	
	 	 	imshow(win_imageA,imageA );
	 	 	imshow(win_imageB,imageB );
	 	 	imshow(win_intermediate,fra );
	 	 	waitKey(70);
	 		}
	 
	 
					

}

