#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
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

//function to get the size of intermediate image
Mat create(Mat src,Mat tran,int& x_m,int& y_m ){
	int row,col;
	int x_mx,x_min;
	float a,b,c;
	
	a=tran.at<float>(0,0);
	b=tran.at<float>(0,1);
	c=tran.at<float>(0,2);

	//cout<<a<<" "<<b<<" "<<c<<endl;
	
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
	//cout<<a<<" "<<b<<" "<<c<<endl;

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

	//cout<<src.rows<<" "<<src.cols<<" "<<row<<" "<<col<<"\n";
	Mat dst=Mat::zeros(row,col,CV_8U);
	return dst;

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

 	for(int i=0;i<2;i++){
 		for(int j=0;j<3;j++){
			res.at<float>(i,j)=con.at<float>(3*i+j);
			//cout<<res.at<float>(i,j)<<" ";
 			}
 			//cout<<endl;
 		}
	return res;
}

//returns image of the input point.
Mat trans(int x,int y,Mat tran){
	Mat m=Mat::ones(3,1,CV_32F);
	m.at<float>(0,0)=x;
	m.at<float>(1,0)=y;
	Mat n=tran*m;
	return n; 
}

//return the frame at given alpha .
Mat frame(Mat src,double alp, Mat tran){

	int gap=20;
	double x1=gap,y1=gap,
		x2=gap,y2=src.cols-gap,
		x3=src.rows-gap,y3=gap,
		x4=src.rows-gap,y4=src.cols-gap,
		x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_;


	Mat n=trans(x1,y1,tran);
	x1_=n.at<float>(0,0);	
	y1_=n.at<float>(1,0);	

	n=trans(x2,y2,tran);
	x2_=n.at<float>(0,0);	
	y2_=n.at<float>(1,0);	

	n=trans(x3,y3,tran);
	x3_=n.at<float>(0,0);	
	y3_=n.at<float>(1,0);	

	n=trans(x4,y4,tran);
	x4_=n.at<float>(0,0);	
	y4_=n.at<float>(1,0);	

	x1_= x1_*alp +x1*(1-alp);
	y1_= y1_*alp +y1*(1-alp);

	x2_= x2_*alp +x2*(1-alp);
	y2_= y2_*alp +y2*(1-alp);

	x3_= x3_*alp +x3*(1-alp);
	y3_= y3_*alp +y3*(1-alp);

	x4_= x4_*alp +x4*(1-alp);	
	y4_= y4_*alp +y4*(1-alp);

	Mat T=trans_mat(x1,y1,x2,y2,x3,y3,x1_,y1_,x2_,y2_,x3_,y3_);

	Mat T_inv=T.inv();

	int y_m,x_m,f=tran.at<float>(1,2),c=tran.at<float>(0,2);
	Mat fra=create(src,T,x_m,y_m);


	/*  i,j of dest and i1,j1 of src */
 	for(int i=0;i<fra.rows;i++){
 		for(int j=0;j<fra.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			Mat m=Mat::ones(3,1,CV_32F);
 			m.at<float>(0,0)=(x_m<0)?i+(x_m):i;
 			m.at<float>(1,0)=(y_m<0)?j+(y_m):j;

 			Mat n= T_inv * m;
 			if(n.at<float>(0,0) >=0 && n.at<float>(0,0) <src.rows && n.at<float>(1,0) >=0 && n.at<float>(1,0)<src.cols ){
 				int i1=n.at<float>(0,0);
 				int j1=n.at<float>(1,0);
 				fra.at<uchar>(i,j)=bilinear(i1,j1,src);
 			}
 		}
 	}

 	return fra;

}

int main(int argc, char** argv){
	
	String win_image = "image"; 
	namedWindow(win_image); 
	
	String win_image_r = " Result image "; 
	namedWindow(win_image_r); 

	String win_intermediate = "intermediate"; 
	namedWindow(win_intermediate); 
	
	string path;
	int num;

	cout<<"Enter the imageA path -  Number of intermediate images\n";
 	cin>>path>>num;
 	cout<<"Enter the Affine transformation 2*3 ";

 	Mat image=imread(path),g_image;
 	if (image.empty()){
	  cout << "Could not open or find the image" << endl;
	  return 0;
	 }
	
	cvtColor( image, g_image, CV_BGR2GRAY );

 	Mat T,res,fra;
	T=Mat::zeros(2,3,CV_32F);
/* 	int num,a,b,c,d,e,f;
 	Point center = Point(0,0);
	Mat rot_mat = getRotationMatrix2D( center,30,1 );
	//print_ma(rot_mat);
	double thet=30*M_PI/180;
	cout<<"trans "<<endl;
	Mat T1=Mat::eye(2,3,CV_32F);
	T1.at<float>(0,0)=(cos(thet)) ;
	T1.at<float>(0,1)=(-sin(thet));
	T1.at<float>(1,0)=(sin(thet));
	T1.at<float>(1,1)=(cos(thet));
	print_ma(T1);
*/

	 	cin>>T.at<float>(0,0)>>T.at<float>(0,1)>>T.at<float>(0,2);
	 	cin>>T.at<float>(1,0)>>T.at<float>(1,1)>>T.at<float>(1,2);

		warpAffine(g_image,res,T, g_image.size() );
	 	/*imshow(windowName_,res );
	 	waitKey(0);*/

	 	double alp=0;
	 	for(int i=0;i<=num;i++){
	 		alp+=1.0/(num+1);
	 		fra=frame(g_image,alp,T);
	 		//cout<<"rows-cols"<<fra.rows<<" "<<fra.cols<<" "<<g_image.rows<<" "<<g_image.cols<<"\n";
	 	 	imshow(win_image,g_image );
	 	 	imshow(win_image_r,res );
	 	 	imshow(win_intermediate,fra );
	 	 	waitKey(300);
	 	}
	 	
					

}