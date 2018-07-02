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
 

uchar bilinear_util(int p1,int p2,double alp,double bet,Mat sr){           
		return round( (1-alp)*(1-bet)*sr.at<uchar>(p1+1,p2+1) + (alp)*(1-bet)*sr.at<uchar>(p1+2,p2+1)
			+(1-alp)*(bet)*sr.at<uchar>(p1+1,p2+2)+(alp)*(bet)*sr.at<uchar>(p1+2,p2+2) );
}

uchar bilinear(double i1,double j1,Mat sr){
	int rows=sr.rows;
	int cols=sr.cols;

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

void get_points(int& x1, int& y1,int& x2, int& y2,int& x3, int& y3,Vec6f t ){
        x1 =cvRound(t[0]);
        y1=cvRound(t[1]);
        x2 =cvRound(t[2]);
        y2=cvRound(t[3]);
        x3 =cvRound(t[4]);
        y3=cvRound(t[5]);
}     

//to find euclidean distance 
double dist(float x1,float y1,float x2,float y2){
	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

//to calculate the area of a triangle.
double area(int x1,int y1,int x2,int y2,int x3,int y3){
	return abs( x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2) )/2.0 ;
}

//to check wether a point is inside or outside the triangle based on sum of areas .
bool inside(int x,int y,int x1,int y1,double x2,int y2,int x3,int y3){
	return ( area(x,y,x1,y1,x2,y2) + area(x,y,x3,y3,x2,y2) + area(x,y,x1,y1,x3,y3) - area(x3,y3,x1,y1,x2,y2) < 0.0001 ) ;  
}

//to get intermediate images delaunauy triangulation
vector<Vec6f> get_intermediate_triangulation(int rows,int cols, Mat img,Mat img2,vector<Point> points_A,Point** mapping,Point** inv_mapping,Subdiv2D& subdiv,double alp){
	
	int x1,y1,x1_,y1_;
	int i_x1,i_y1;

 	for( vector<Point>::iterator it = points_A.begin(); it != points_A.end(); it++){
	        x1=(*it).x;
	        y1=(*it).y;
	        x1_=mapping[x1][y1].x; y1_=mapping[x1][y1].y;

	        i_x1= x1_*alp +x1*(1-alp);
			i_y1= y1_*alp +y1*(1-alp);

			inv_mapping[i_x1][i_y1]=Point(x1,y1);
			subdiv.insert(Point(i_x1,i_y1));

        }

    vector<Vec6f> triangleList;
    vector<Vec6f> resList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Rect rect(0,0,cols,rows);
 	
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

//to get the affine transfromation applied between given pair of points.
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
	return con;
}

//return the frame at given alpha .

Mat frame(Mat src,Mat dst,double alp, Point** mapping, vector<Point> points_A ){

	int rows=round(src.rows*(1-alp)+dst.rows*alp);
	int cols=round(src.cols*(1-alp)+dst.cols*alp);

	Mat fra=Mat::zeros(rows,cols,CV_8U);

	Mat a_inv=Mat::zeros(rows,cols,CV_8U);
	Mat b_inv=Mat::zeros(rows,cols,CV_8U);

   /* 
	Some windows for testing the code.
   String win_imageA_in = "imageA_inv"; 
	namedWindow(win_imageA_in); 
	
    String win_imageB_in = "imagB_inv"; 
	namedWindow(win_imageB_in); 

    String win_imagefr = "imagfr"; 
	namedWindow(win_imagefr); 
	
	*/
	// Rectangle to be used with Subdiv2D
    Size size = fra.size();
    
    Rect rect(0, 0, size.width, size.height);

 	// Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    	

 	Point** inv_mapping=new Point*[cols];
	for(int i = 0; i < cols; ++i)
	   	inv_mapping[i] = new Point[rows];

	Mat src_clo=src.clone();   
	Mat dst_clo=dst.clone();   

	vector<Vec6f> triangleList=get_intermediate_triangulation(rows,cols,src_clo,dst_clo,points_A,mapping,inv_mapping,subdiv,alp);
	/*imshow(win_imageA_in,src_clo);
	imshow(win_imageB_in,dst_clo);
	waitKey(0);
*/
	int x1,y1,x2,y2,x3,y3,x1_,y1_,x2_,y2_,x3_,y3_;
	int i_x1,i_y1,i_x2,i_y2,i_x3,i_y3;

    for( size_t i = 0; i < triangleList.size(); i++ ){
    	Vec6f trian=triangleList[i];
    	get_points(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,trian);
	
		x1=inv_mapping[i_x1][i_y1].x;
		y1=inv_mapping[i_x1][i_y1].y;


		x2=inv_mapping[i_x2][i_y2].x;
		y2=inv_mapping[i_x2][i_y2].y;

		x3=inv_mapping[i_x3][i_y3].x;
		y3=inv_mapping[i_x3][i_y3].y;
	

    	x1_=mapping[x1][y1].x;
    	y1_=mapping[x1][y1].y;

    	x2_=mapping[x2][y2].x;
    	y2_=mapping[x2][y2].y;

    	x3_=mapping[x3][y3].x;
    	y3_=mapping[x3][y3].y;


		Point fp(i_x1,i_y1);
	
		Mat inv_A=trans_mat(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,x1,y1,x2,y2,x3,y3);
		Mat inv_B=trans_mat(i_x1,i_y1,i_x2,i_y2,i_x3,i_y3,x1_,y1_,x2_,y2_,x3_,y3_);

		int x_max= max(i_x1,max(i_x2,i_x3)) ;
		int x_min= min(i_x1,min(i_x2,i_x3)) ;

		int y_max= max(i_y1,max(i_y2,i_y3)) ;
		int y_min= min(i_y1,min(i_y2,i_y3)) ;

		//cout<<"imB "<<x11<<" "<<y11<<" "<<x4_<<" "<<y4_<<" "<<x1_<<","<<y1_<<" "<<x2_<<","<<y2_<<" "<<x3_<<","<<y3_<<" \n";
		/*cout<<"imB "<<x11<<" "<<y11<<" "<<x4_<<" "<<y4_<<" "<<x1_<<","<<y1_<<" "<<x2_<<","<<y2_<<" "<<x3_<<","<<y3_<<" \n";
		cout<<"inter "<<i_x4<<" "<<i_y4<<" "<<i_x1<<","<<i_y1<<" "<<i_x2<<","<<i_y2<<" "<<i_x3<<","<<i_y3<<" \n";
*/
		//cout<<i_x1<<" "<<i_y1<<" "<<i_x2<<" "<<i_y2<<" "<<i_x3<<" "<<i_y3<<" "<<endl;
//		cout<<"sqr "<<x_min<<" "<<x_max<< "    "<<y_min<<" "<<y_max<<endl; 

		for(int i=y_min;i<=y_max;i++){
			for(int j=x_min;j<=x_max;j++){
				if(inside(j,i,i_x1,i_y1,i_x2,i_y2,i_x3,i_y3)){

					double j1= inv_A.at<float>(0,0)* j +inv_A.at<float>(1,0)* i+inv_A.at<float>(2,0); 			
					double i1= inv_A.at<float>(3,0)* j +inv_A.at<float>(4,0)* i+inv_A.at<float>(5,0); 			
			 		a_inv.at<uchar>(i,j)=bilinear(i1,j1,src);      //finding the values for original image from intensities of distorted ones.
			 		
			 		//cout<<i<<" "<<j<<" -- "<<i1<<" "<<j1<<endl;

			 		j1= inv_B.at<float>(0,0)* j +inv_B.at<float>(1,0)* i+inv_B.at<float>(2,0); 			
					i1= inv_B.at<float>(3,0)* j +inv_B.at<float>(4,0)* i+inv_B.at<float>(5,0); 			
			 		b_inv.at<uchar>(i,j)= bilinear(i1,j1,dst) ;      //finding the values for original image from intensities of distorted ones.
			 		
			 		fra.at<uchar>(i,j)= a_inv.at<uchar>(i,j)*(1-alp)+b_inv.at<uchar>(i,j)*alp ;      //finding the values for original image from intensities of distorted ones.
				}
			}
		}
		
/*
		Some windows for testing the code.
  
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
    /* 
	imshow(win_imageA_in,a_inv );
	imshow(win_imageB_in,b_inv );*/
	waitKey(70);	
 	return fra;

}


int main(int argc, char** argv){

	String win_imageA = "imageA"; 
	namedWindow(win_imageA); 
	
	String win_imageB = " imageB "; 
	namedWindow(win_imageB); 

	String win_intermediate = "intermediate"; 
	namedWindow(win_intermediate); 
	
 	Mat fra;
 	int num=1;
	
	ifstream fi("inp.txt");

	string pathA,textA,pathB,textB;
 	cout<<"Enter the imageA path - imageA points.txt - imageB path - imageB points.txt - Number of intermediate images\n";
 	fi>>pathA>>textA>>pathB>>textB>>num;

 	ifstream fin;
 	fin.open(textA.c_str());
	//fin.open("hillary_clinton.jpg.txt");

	ifstream fin1;
	fin1.open(textB.c_str());
	//fin1.open("ted_cruz.jpg1.txt");

 	Mat imageA=imread(pathA);
 	if (imageA.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	cvtColor( imageA, imageA, CV_BGR2GRAY );


 	Mat imageB=imread(pathB);
 	if (imageB.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	cvtColor( imageB, imageB, CV_BGR2GRAY );

 	Point** mapping=new Point*[imageA.cols];
	for(int i = 0; i < imageA.cols; ++i)
    	mapping[i] = new Point[imageA.rows];

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

	    double alp=0;
	 	for(int i=0;i<=num+1;i++){
	 		ofstream fout;
	    	stringstream out;
	    	out<<"frame"<<i<<".jpg";
	 		//fout.open( to_string(i)+".png");
	 		fra=frame(imageA,imageB,alp,mapping,points_A);
	 		alp+=1.0/(num+1);
	 		imwrite(out.str(),fra);
	 		imshow(win_imageA,imageA );
	 	 	imshow(win_imageB,imageB );
	 	 	imshow(win_intermediate,fra );
	 	 	waitKey(100);
	 		}
}

