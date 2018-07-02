#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;
enum interpol { 
	LINEAR,
	NEAREST
	};

double rmse(Mat a,Mat b){
	if(a.rows!=b.rows && a.cols!=b.cols)  return 0;
	double res=0;
	for(int i=0;i<a.rows;i++){
		for(int j=0;j<a.cols;j++){
			res+= ( a.at<uchar>(i,j)-b.at<uchar>(i,j) )*( a.at<uchar>(i,j)-b.at<uchar>(i,j) );
		}
	}
	res/=(a.rows*a.cols);
	return sqrt(res);
 } 

uchar bilinear_util(int p1,int p2,double alp,double bet,Mat sr){
		return round( (1-alp)*(1-bet)*sr.at<uchar>(p1+1,p2+1) + (alp)*(1-bet)*sr.at<uchar>(p1+2,p2+1)
			+(1-alp)*(bet)*sr.at<uchar>(p1+1,p2+2)+(alp)*(bet)*sr.at<uchar>(p1+2,p2+2) );
}

/* some genral functions for intensity assignment */
uchar nearest(double i1,double j1,Mat src){
	int i_s;
	if(i1<0){
		i_s=0;
	}else{
		 i_s=(int(i1)+1);
	}
	int j_s;
	if(j1<0){
		j_s=0;
	}else{
		 j_s=(int(j1)+1);
	}
	return src.at<uchar>(i_s,j_s);	
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

Mat create(Mat src,Mat tran){
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

	cout<<src.rows<<" "<<src.cols<<" ";
	cout<<row<<" "<<col<<"\n";
	Mat dst;
	if(row>0 && col>0 ) dst=Mat::zeros(row,col,CV_8U);
	//cout<<"sfa\n";
	
	return dst;

}

Mat m_resize(Mat src,double fx,double fy,int interpol=LINEAR ){
/*	 Mat n;
	 if(src.empty() && (fy*src.cols)>0 && (fx*src.rows)>0){
	 	return n;
	 }
	
	Mat dst(round(fx*src.rows),round(fy*src.cols),CV_8U);
*/	
	Mat T=Mat::eye(3,3,CV_32F);
	T.at<float>(0,0)=fx;
	T.at<float>(1,1)=fy;

	Mat dst=create(src,T);

	if(interpol==NEAREST){
	 	/*  i,j of dest and i1,j1 of src */
	 	for(int i=0;i<dst.rows;i++){
	 		for(int j=0;j<dst.cols;j++){
	 			//cout<<i<<"   "<<j<<endl;
				double step=1.0/fx;
	 			double i1=(-1+1.0/fx*(i+1) );
	 			i1-=step/2;
	 			step=1.0/fy;
	 			double j1=(-1+1.0/fy*(j+1) ) ;
	 			j1-=step/2;
	 			dst.at<uchar>(i,j)=nearest(i1,j1,src);
			}
	 	}
	 }
	 else{

		Mat pad_src=Mat::zeros(src.rows+2,src.cols+2,CV_8U);      //padded source for bilinear interpolation.
		for(int i=0;i<src.rows;i++){
		 	for(int j=0;j<src.cols;j++){
			 		pad_src.at<uchar>(i+1,j+1) = src.at<uchar>(i,j);
			 	}
		 }
			
	 	for(int i=0;i<dst.rows;i++){
	 		for(int j=0;j<dst.cols;j++){
	 		////cout<<i<<"  "<<j<<endl;
	 		double step=1.0/fx;
			double i1=(-1+1.0/fx*(i+1) );
			i1-=step/2;
			
			step=1.0/fy;
			double j1=(-1+1.0/fy*(j+1) ) ;
			j1-=step/2;

			dst.at<uchar>(i,j)=bilinear(i1,j1,pad_src);
			}
	 	}
	 	////cout<<dst.rows<<" "<<dst.cols<<endl;
	 }
	 return dst;

}


Mat translate(float dx,float dy,Mat src){
	//Mat dst=Mat::zeros(src.rows+dx ,src.cols+dy,CV_8U);

	Mat T=Mat::eye(3,3,CV_32F);
	T.at<float>(0,2)=dx;
	T.at<float>(1,2)=dy;

	Mat dst=create(src,T);
	
	Mat T_inv=Mat::eye(3,3,CV_32F);
	T_inv.at<float>(0,2)=-dx;
	T_inv.at<float>(1,2)=-dy;

	/*  i,j of dest and i1,j1 of src */
	 	for(int i=0;i<dst.rows;i++){
	 		for(int j=0;j<dst.cols;j++){
	 			//cout<<i<<"   "<<j<<endl;
	 			Mat m=Mat::ones(3,1,CV_32F);
	 			m.at<float>(0,0)=i;
	 			m.at<float>(1,0)=j;

	 			Mat n= T_inv * m;
	 			if(n.at<float>(0,0)>=0 && n.at<float>(0,0)<src.rows && n.at<float>(1,0)>=0 && n.at<float>(1,0)<src.cols ){
	 				int i1=n.at<float>(0,0);
	 				int j1=n.at<float>(1,0);
	 				dst.at<uchar>(i,j)=src.at<uchar>(i1,j1);
	 			}
	 		}
	 	}
	 	return dst;
}

Mat rotate(double thet,Mat src,int interpol=LINEAR){
	
	cout<<thet<<endl;
	thet=-thet/180*M_PI;
	
	int dig=sqrt(src.rows*src.rows+src.cols*src.cols);
	double alpha=atan((double)src.rows/src.cols),ang;

	cout<<alpha/M_PI*180<<endl;
	int rows,cols;
	ang=(thet+alpha > 2*M_PI ) ? thet+alpha-2*M_PI :thet+alpha ;  
	ang=(ang < 0 ) ? ang+2*M_PI :ang;  
	
	cout<<ang/M_PI*180<<endl;
	
	if( ang>=0 && ang<=M_PI/2 ){
		cols = dig*cos(ang);
		rows = dig*sin(ang);
	}
    else if( ang>=M_PI/2 && ang<=M_PI ){
		cols = dig*cos(M_PI-ang);
		rows = dig*sin(M_PI-ang);
	}
    else if( ang>=M_PI && ang<=3*M_PI/2 ){
		cols = dig*cos(ang-M_PI);
		rows = dig*sin(ang-M_PI);
	}
    else if( ang>=3*M_PI/2 && ang<=2*M_PI ){
		cols = dig*cos(2*M_PI-ang);
		rows = dig*sin(2*M_PI-ang);
	}
	cout<<rows<<" - "<<cols<<endl;

	ang=(-thet+alpha > 2*M_PI ) ? -thet+alpha-2*M_PI :-thet+alpha ;  
	ang=(ang < 0 ) ? ang+2*M_PI :ang ;  
	
	cout<<ang/M_PI*180<<endl;
	
	if( ang>=0 && ang<=M_PI/2 ){
		cols = (cols<dig*cos(ang)) ? dig*cos(ang) :cols;
		rows = (rows<dig*sin(ang)) ? dig*sin(ang):rows;
	}
    else if( ang>=M_PI/2 && ang<=M_PI ){
		cols = (cols<dig*cos(M_PI-ang)) ? dig*cos(M_PI-ang):cols;
		rows = (rows<dig*sin(M_PI-ang)) ? dig*sin(M_PI-ang) :rows;
	}
    else if( ang>=M_PI && ang<=3*M_PI/2 ){
		cols = (cols<dig*cos(-M_PI+ang)) ? dig*cos(-M_PI+ang) :cols;
		rows = (rows<dig*sin(ang-M_PI)) ? dig*sin(ang-M_PI):rows;
	}
    else if( ang>=3*M_PI/2 && ang<=2*M_PI ){
		cols = (cols<dig*cos(2*M_PI-ang)) ? dig*cos(2*M_PI-ang):cols ;
		rows = (rows<dig*sin(2*M_PI-ang)) ? dig*sin(2*M_PI-ang):rows;
	}

	cout<<rows<<" - "<<cols<<endl;

	int shx,shy; 
	//Mat dst=Mat::zeros( rows,cols ,CV_8U );
	
	Mat T1=Mat::eye(3,3,CV_32F);
	T1.at<float>(0,0)=(cos(thet)) ;
	T1.at<float>(0,1)=(-sin(thet));
	T1.at<float>(1,0)=(sin(thet));
	T1.at<float>(1,1)=(cos(thet));
	
	Mat dst=create(src,T1);

	shx=dst.rows-src.rows;
	shy=dst.cols-src.cols;

	Mat T_inv=Mat::eye(3,3,CV_32F);
	T_inv.at<float>(0,0)=(cos(thet)) ;
	T_inv.at<float>(0,1)=(sin(thet));
	T_inv.at<float>(1,0)=(-sin(thet));
	T_inv.at<float>(1,1)=(cos(thet));

	Mat T=Mat::eye(3,3,CV_32F);
	T_inv.at<float>(0,0)=(cos(thet)) ;
	T_inv.at<float>(0,1)=(-sin(thet));
	T_inv.at<float>(1,0)=(sin(thet));
	T_inv.at<float>(1,1)=(cos(thet));
	
	Mat m=Mat::ones(3,1,CV_32F);
	m.at<float>(0,0)=src.rows/2;
	m.at<float>(1,0)=src.cols/2;

	int x=15;
	int y=447;
	
	Mat n=T * m;
	int cr1=n.at<float>(0,0),cr2=n.at<float>(0,1);
	
		if(interpol==LINEAR){
		/*  i,j of dest and i1,j1 of src */
		 	for(int i=0;i<dst.rows;i++){
		 		for(int j=0;j<dst.cols;j++){

		 			Mat m=Mat::ones(3,1,CV_32F);
		 			m.at<float>(0,0)=i-src.rows/2-shx/2;
		 			m.at<float>(1,0)=j-src.cols/2-shy/2;

		 			Mat n= T_inv * m;

		 			if(i==x && j==y ){
		 				cout<<n.at<float>(0,0)+cr1<<" "<<n.at<float>(1,0)+cr2<<" ";
		 				cout<<n.at<float>(0,0)+cr1+shx/2<<" "<<n.at<float>(1,0)+cr2+shy/2<<" ";
		 				cout<<i<<"   "<<j<<" \n";
		 				x+=100;
		 				y+=100;
		 			}
		 			if(n.at<float>(0,0) + cr1 >=0 && n.at<float>(0,0) + cr1<src.rows && n.at<float>(1,0) + cr2 >=0 && n.at<float>(1,0) + cr2<src.cols ){
		 				int i1=n.at<float>(0,0)+cr1;
		 				int j1=n.at<float>(1,0)+cr2;
		 				dst.at<uchar>(i,j)=bilinear(i1,j1,src);
		 			}
		 		}
		 	}
		 }else{
		 	/*  i,j of dest and i1,j1 of src */
		 	for(int i=0;i<dst.rows;i++){
		 		for(int j=0;j<dst.cols;j++){
		 			//cout<<i<<"   "<<j<<endl;
		 			Mat m=Mat::ones(3,1,CV_32F);
					m.at<float>(0,0)=i-src.rows/2-shx/2;
		 			m.at<float>(1,0)=j-src.cols/2-shy/2;

		 			Mat n= T_inv * m;
		 			if(n.at<float>(0,0) + cr1 >=0 && n.at<float>(0,0) + cr1<src.rows && n.at<float>(1,0) + cr2 >=0 && n.at<float>(1,0) + cr2<src.cols ){
		 				int i1=n.at<float>(0,0)+cr1;
		 				int j1=n.at<float>(1,0)+cr2;
		 				dst.at<uchar>(i,j)=nearest(i1,j1,src);
		 			}
		 		}
			 }	
		}
		 	return dst;
}

Mat shear(double a,double b,Mat src,int interpol=LINEAR){
	int crx,cry;
	crx=(a>0)?a*src.cols:-1*a*src.cols;
	cry=(b>0)?b*src.rows:-1*b*src.rows;
	
	Mat dst=Mat::zeros(src.rows+ crx +5 , src.cols+cry +5 ,CV_8U);
	
	Mat T_inv=Mat::eye(3,3,CV_32F);
	T_inv.at<float>(0,0)=1/(1-a*b) ;
	T_inv.at<float>(0,1)=-a/(1-a*b);
	T_inv.at<float>(1,0)=-b/(1-a*b);
	T_inv.at<float>(1,1)=1/(1-a*b);

	if(interpol==LINEAR){
		/*  i,j of dest and i1,j1 of src */
		 	for(int i=0;i<dst.rows;i++){
		 		for(int j=0;j<dst.cols;j++){
		 			//cout<<i<<"   "<<j<<endl;
		 			Mat m=Mat::ones(3,1,CV_32F);
		 			m.at<float>(0,0)=(a<0)?i-crx:i;
		 			m.at<float>(1,0)=(a<0)?j-cry:j;

		 			Mat n= T_inv * m;
		 			if(n.at<float>(0,0) >=0 && n.at<float>(0,0) <src.rows && n.at<float>(1,0) >=0 && n.at<float>(1,0)<src.cols ){
		 				int i1=n.at<float>(0,0);
		 				int j1=n.at<float>(1,0);
		 				dst.at<uchar>(i,j)=bilinear(i1,j1,src);
		 			}
		 		}
		 	}
		 }else{
		 	/*  i,j of dest and i1,j1 of src */
		 	for(int i=0;i<dst.rows;i++){
		 		for(int j=0;j<dst.cols;j++){
		 			//cout<<i<<"   "<<j<<endl;
		 			Mat m=Mat::ones(3,1,CV_32F);

		 			m.at<float>(0,0)=(a<0)?i-crx:i;
		 			m.at<float>(1,0)=(a<0)?j-cry:j;

		 			Mat n= T_inv * m;
		 			if(n.at<float>(0,0)  >=0 && n.at<float>(0,0) <src.rows && n.at<float>(1,0)>=0 && n.at<float>(1,0) <src.cols ){
		 				int i1=n.at<float>(0,0);
		 				int j1=n.at<float>(1,0);
		 				dst.at<uchar>(i,j)=nearest(i1,j1,src);
		 			}
		 		}
			 }	
		}
		 	return dst;
}


Mat negative(Mat src){
	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			dst.at<uchar>(i,j)=255-src.at<uchar>(i,j);
 			}
 		}
 	 	return dst;
}

Mat log_t(double c, Mat src){
	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			dst.at<uchar>(i,j)= c*log(1+src.at<uchar>(i,j));      //log transform found 75 to be best value.
 			}
 		}
 	 	return dst;
}


Mat gamma_t(double gam,double c,Mat src){
	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			dst.at<uchar>(i,j)= c*pow(src.at<uchar>(i,j),gam);      //log transform found 75 to be best value.
 			}
 		}
 	 	return dst;
}


Mat piecewise_t(int r1,int r2,int s1,int s2,Mat src){
	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			uchar r=src.at<uchar>(i,j);
 			if(r<=r1){
 				if(r1==0) dst.at<uchar>(i,j)=s1;
 				else
 					dst.at<uchar>(i,j)= (s1/r1)*r ;  //log transform found 75 to be best value.

 			}else if(r>=r2){
 				if(r2==255) dst.at<uchar>(i,j)=255;

 				dst.at<uchar>(i,j)= (255-s2)/(255-r2)*(r-r2) +255 ;      //log transform found 75 to be best value.
  			}else
 				dst.at<uchar>(i,j)= (s2-s1)/(r2-r1)*(r-r1) + s1 ;      //log transform found 75 to be best value.
 			}
 		}
 	 	return dst;
}

Mat bit_plane(int pla,Mat src){
	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
 	pla--;

 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			//cout<<i<<"   "<<j<<endl;
 			uchar r=src.at<uchar>(i,j);
 			dst.at<uchar>(i,j)= (r & (1<<pla) ) ;      //log transform found 75 to be best value.
 			}
 		}
 	 	return dst;
}

Mat hist_eqal(Mat src){                    

	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
	double pro[256]={};

	for(int i=0;i<src.rows;i++){
 		for(int j=0;j<src.cols;j++){
			pro[src.at<uchar>(i,j)]++;      //log transform found 75 to be best value.
 			}
 		}
 	

	double sum=0;
	for( int i=0;i<256;i++){
			pro[i]=pro[i]/(src.cols*src.rows);      //normalizing the pro values.
			sum+=pro[i];
			pro[i]=sum;
 	}
	
	for(int i=0;i<256;i++){
			pro[i]*=255;
 	}
	
 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			dst.at<uchar>(i,j)= round(pro[src.at<uchar>(i,j)]);      //log transform found 75 to be best value.
 			}
 		}
	return dst;
}


Mat hist_match(Mat src,Mat B){                    

	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
	double pro_s[256]={} , pro_b[256]={};

	for(int i=0;i<src.rows;i++){
 		for(int j=0;j<src.cols;j++){
			pro_s[src.at<uchar>(i,j)]++;      //log transform found 75 to be best value.
 			}
 		}

	double sum=0;
	for( int i=0;i<256;i++){
			pro_s[i]=pro_s[i]/(src.cols*src.rows);      //normalizing the pro values.
			sum+=pro_s[i];
			pro_s[i]=sum;
 	}
	for(int i=0;i<256;i++){
			pro_s[i]*=255;
 	}


	for(int i=0;i<B.rows;i++){
 		for(int j=0;j<B.cols;j++){
			pro_b[B.at<uchar>(i,j)]++;      //log transform found 75 to be best value.
 			}
 		}

	sum=0;
	for( int i=0;i<256;i++){
			pro_b[i]=pro_b[i]/(B.cols*B.rows);      //normalizing the pro values.
			sum+=pro_b[i];
			pro_b[i]=sum;
 	}

	for(int i=0;i<256;i++){
			pro_b[i]*=255;

 	}

 	for(int i=0,j=0;i<256 && j<256;i++){             // i fro src and j for B.
 		double dif=abs(pro_s[i]-pro_b[j]);
 		double old_dif=280;

 		while(old_dif >= dif){
 				old_dif=dif;
 				dif=abs(pro_s[i]-pro_b[++j]);
 		}
 		j--;
 		pro_s[i]=j;
 	}

 	for(int i=0;i<dst.rows;i++){
 		for(int j=0;j<dst.cols;j++){
 			
 			dst.at<uchar>(i,j)= pro_s[src.at<uchar>(i,j)];      //log transform found 75 to be best value.
 			}
 		}
	return dst;
}


Mat sw_adap_hist_eqal(int win_rows,int win_cols,Mat src){                    

	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
	Mat pad_src=Mat::zeros(src.rows+ (win_rows/2)*2 ,src.cols+(win_cols/2)*2 ,CV_8U);
	
	for(int i=0;i<src.rows;i++){
 		for(int j=0;j<src.cols;j++){
			pad_src.at<uchar>(i+win_rows/2,j+win_cols/2)= src.at<uchar>(i,j) ;      //log transform found 75 to be best value.
 			}
 	}
 	String windowName2 = "opencv g_image adp eq pad_src "; 
	 namedWindow(windowName2,WINDOW_NORMAL); 
	 imshow(windowName2, pad_src); 

	 waitKey(1000);
 	for(int i=0;i<win_rows/2;i++){
 		for(int j=0;j<pad_src.cols;j++){
 			if( j<win_cols/2 ){
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>(i+2*(win_rows/2-i)-1 ,j+2*(win_cols/2-j)-1) ;      //log transform found 75 to be best value.
 			
 			}
 			else if( j > pad_src.cols-win_cols/2-1 ){
 					int ry=j-(pad_src.cols-win_cols/2-1);
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i+ 2*(win_rows/2-i)-1 ,j-2*ry+1 ) ;      //log transform found 75 to be best value.
 			} 
 			else{	
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i+2*(win_rows/2-i)-1,j ) ;      //log transform found 75 to be best value.
 			}
 		}
 	}

 	for(int i=pad_src.rows-1,k=0; k < win_rows/2 ;i--,k++){
		
		int rx=i-(pad_src.rows-win_rows/2-1);
 		
 		for(int j=0;j<pad_src.cols;j++){
 			if( j<win_cols/2 ){
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>(i-2*rx+1,j+2*(win_rows/2-j)-1) ;      //log transform found 75 to be best value.
 			
 			}
 			else if( j > pad_src.cols-win_cols/2-1 ){
 					int ry=j-(pad_src.cols-win_cols/2-1);
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i-2*rx+1 ,j-2*ry+1 ) ;      //log transform found 75 to be best value.
 			} 
 			else{
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i-2*rx+1,j ) ;      //log transform found 75 to be best value.
 			}
 		}
 	}


 	for(int j=0;j<win_cols/2;j++){
 		for(int i=win_rows/2;i< pad_src.rows-win_rows/2 ;i++){
 				pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i,j+2*(win_rows/2-j)-1 ) ;      //log transform found 75 to be best value.
 		}
 	}

 	for(int j= pad_src.cols-1,k=0; k < win_cols/2 ;j--,k++){
		int ry=j-(pad_src.cols-win_cols/2-1);

 		for(int i=win_rows/2;i< pad_src.rows-win_rows/2 ;i++){
 				pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i,j-2*ry+1 ) ;      //log transform found 75 to be best value.
 			}
 	}
	imshow(windowName2, pad_src); 

 	/*  Matrix to preserve old values . */
 	Mat old_pad_src=pad_src.clone();         

	double pro1[256]={},pro[256]={},temp_pro[256]={};
	int low=256,high=0;
	
	for(int i1=0;i1<win_rows;i1++){
		 		for(int j1=0;j1<win_cols;j1++){
					pro[old_pad_src.at<uchar>(i1,j1)]++;      //precomputed values value.
					temp_pro[old_pad_src.at<uchar>(i1,j1)]++;      //precomputed values value.
					pro[old_pad_src.at<uchar>(i1,j1)]++;
					if(high < pad_src.at<uchar>(i1,j1) ) high=pad_src.at<uchar>(i1,j1);

		 }
	}

	int step=4;
	for(int i=step/2;i<src.rows;i+=step){
		for(int j=step/2;j<src.cols;j+=step){
			double sum=0;
			cout<<(int)src.at<uchar>(i,j)<<endl;
			for( int k=0;k <= 255; k++){
					sum+=pro1[k];
					pro1[k]=sum;
		 	}
		 	for(int i1=i-step/2;i1<i+step/2;i1++){
			 	for(int j1=j-step/2;j1<j+step/2;j1++){
					if(i1<dst.rows && j1<dst.cols )
					dst.at<uchar>(i1,j1)=round(high*pro1[old_pad_src.at<uchar>(i1,j1)]/(win_rows * win_cols));
			 	}
		 	}

			for(int i1=0;i1<win_rows;i1++){
				pro[old_pad_src.at<uchar>(i1,j)]--;      //updating precomputed values value.
				pro[old_pad_src.at<uchar>(i1,j+win_cols)]++;      //updating precomputed values value.
			}
		}

 		for(int j1=0;j1<win_cols;j1++){
			temp_pro[old_pad_src.at<uchar>(i,j1)]--;      //updating precomputed values value.
			temp_pro[old_pad_src.at<uchar>(i+win_rows,j1)]++;      //updating precomputed values value.
		}

		
 		for(int j1=0;j1<256;j1++){
			pro[j1]=temp_pro[j1];
			}
		

	}

	return dst;
}


Mat adap_hist_eqal(int win_rows,int win_cols,Mat src){                    

	Mat dst=Mat::zeros(src.rows,src.cols,CV_8U);
	int corrx=(src.rows % win_rows==0)?0:(win_rows - src.rows % win_rows);
	int corry=(src.cols % win_cols==0)?0:(win_cols - src.cols % win_cols);

	Mat pad_src=Mat::zeros(src.rows+ corrx ,src.cols+corry ,CV_8U);
	
	for(int i=0;i<src.rows;i++){
 		for(int j=0;j<src.cols;j++){
			pad_src.at<uchar>(i,j)= src.at<uchar>(i,j) ;      //log transform found 75 to be best value.
 			}
 	}

 	for(int i=src.rows;i< src.rows+corrx ;i++){
		
		int rx=i-(src.rows-1);
 		
 		for(int j=0;j< src.cols+corry;j++){
 			if( j > src.cols-1 ){
 					int ry=j-(src.cols-1);
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i-2*rx+1 ,j-2*ry+1 ) ;      //log transform found 75 to be best value.
 			} 
 			else{	
 					pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i-2*rx+1 ,j ) ;      //log transform found 75 to be best value.
 			}
 		}
 	}

 	for(int j= src.cols ; j < src.cols+corry ;j++ ){
		int ry=j-(src.cols-1);

 		for(int i=0;i< src.rows ;i++){
 				pad_src.at<uchar>(i,j)= pad_src.at<uchar>( i,j-2*ry+1 ) ;      //log transform found 75 to be best value.
 			}
 	}

	for(int i=0 ;i<pad_src.rows;i+=win_rows){
		for(int j=0 ;j<pad_src.cols;j+=win_cols){
			double sum=0;
			double pro[256]={};
			int low=256,high=0;
			for(int i1=0;i1<win_rows;i1++){
		 		for(int j1=0;j1<win_cols;j1++){
					pro[pad_src.at<uchar>(i+i1,j+j1)]++;      //precomputed values value.
					if(low > pad_src.at<uchar>(i+i1,j+j1) ) low=pad_src.at<uchar>(i+i1,j+j1);
					if(high < pad_src.at<uchar>(i+i1,j+j1) ) high=pad_src.at<uchar>(i+i1,j+j1);
		 			}
			}

			for( int k=0;k <256; k++){
					sum+=pro[k];
					pro[k]=sum;
		 	}

		 	for( int k=0;k <256; k++){
					pro[k]=255*pro[k]/(win_rows * win_cols);
					pro[k]*=(high)/255.0;
		 	}

		 	for(int i1=0;i1<win_rows;i1++){
		 		for(int j1=0;j1<win_cols;j1++){
		 			if( i+i1 < dst.rows && j+j1 < dst.cols )
							dst.at<uchar>(i+i1,j+j1)=round(pro[src.at<uchar>(i+i1,j+j1)]);
		 			}
			}
		}
	}

	return dst;
}


  // x,y's are from the original image whereas x_,y_'s are from distorted ones.
Mat tie_points(float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4,
	float x1_,float y1_,float x2_,float y2_,float x3_,float y3_,float x4_,float y4_,Mat src,int interpol=LINEAR){                            //returns transformation matrix.
	
	Mat t1=Mat::zeros(8,1,CV_32F);
	t1.at<float>(0,0)=x1;
	t1.at<float>(1,0)=y1;
	t1.at<float>(2,0)=x2;
	t1.at<float>(3,0)=y2;
	t1.at<float>(4,0)=x3;
	t1.at<float>(5,0)=y3;
	t1.at<float>(6,0)=x4;
	t1.at<float>(7,0)=y4;
	
	Mat tran1=Mat::zeros(8,8,CV_32F);
	tran1.at<float>(0,0)=x1_;
	tran1.at<float>(0,1)=y1_;
	tran1.at<float>(0,2)=x1_*y1_;
	tran1.at<float>(0,3)=1;
	tran1.at<float>(1,4)=x1_;
	tran1.at<float>(1,5)=y1_;
	tran1.at<float>(1,6)=x1_*y1_;
	tran1.at<float>(1,7)=1;
	tran1.at<float>(2,0)=x2_;
	tran1.at<float>(2,1)=y2_;
	tran1.at<float>(2,2)=x2_*y2_;
	tran1.at<float>(2,3)=1;
	tran1.at<float>(3,4)=x2_;
	tran1.at<float>(3,5)=y2_;
	tran1.at<float>(3,6)=x2_*y2_;
	tran1.at<float>(3,7)=1;
	tran1.at<float>(4,0)=x3_;
	tran1.at<float>(4,1)=y3_;
	tran1.at<float>(4,2)=x3_*y3_;
	tran1.at<float>(4,3)=1;
	tran1.at<float>(5,4)=x3_;
	tran1.at<float>(5,5)=y3_;
	tran1.at<float>(5,6)=x3_*y3_;
	tran1.at<float>(5,7)=1;
	tran1.at<float>(6,0)=x4_;
	tran1.at<float>(6,1)=y4_;
	tran1.at<float>(6,2)=x4_*y4_;
	tran1.at<float>(6,3)=1;
	tran1.at<float>(7,4)=x4_;
	tran1.at<float>(7,5)=y4_;
	tran1.at<float>(7,6)=x4_*y4_;

	Mat con1= tran1.inv()*t1;

	int i=0,j=0;
	double d1= con1.at<float>(0,0)* i +con1.at<float>(1,0)* j+con1.at<float>(2,0)* i*j+con1.at<float>(3,0); 			
	double f1= con1.at<float>(4,0)* i +con1.at<float>(5,0)* j+con1.at<float>(6,0)* i*j+con1.at<float>(7,0); 			
	
	i=src.rows-1,j=src.cols-1;

	double d2= con1.at<float>(0,0)* i +con1.at<float>(1,0)* j+con1.at<float>(2,0)* i*j+con1.at<float>(3,0); 			
	double f2= con1.at<float>(4,0)* i +con1.at<float>(5,0)* j+con1.at<float>(6,0)* i*j+con1.at<float>(7,0); 	

	Mat dst;
	/*if(d2-d1 >src.rows && f2-f1>src.cols)
		dst=Mat::zeros(d2-d1,f2-f1,CV_8U);
	else*/
		dst=Mat::zeros(src.rows,src.cols,CV_8U);

		
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

	if(interpol==LINEAR){
	 	for(int i=0;i<dst.rows;i++){
	 		for(int j=0;j<dst.cols;j++){
				double i1= con.at<float>(0,0)* i +con.at<float>(1,0)* j+con.at<float>(2,0)* i*j+con.at<float>(3,0); 			
				double j1= con.at<float>(4,0)* i +con.at<float>(5,0)* j+con.at<float>(6,0)* i*j+con.at<float>(7,0); 			
	 			dst.at<uchar>(i,j)= bilinear(i1,j1,src);      //log transform found 75 to be best value.
	 			}
	 		}
	}
	else{
		for(int i=0;i<dst.rows;i++){
	 		for(int j=0;j<dst.cols;j++){
				double i1= con.at<float>(0,0)* i +con.at<float>(1,0)* j+con.at<float>(2,0)* i*j+con.at<float>(3,0); 			
				double j1= con.at<float>(4,0)* i +con.at<float>(5,0)* j+con.at<float>(6,0)* i*j+con.at<float>(7,0); 			
	 			dst.at<uchar>(i,j)= nearest(i1,j1,src) ;      //log transform found 75 to be best value.
	 			}
	 		}

	} 	 	
	return dst;
}


int main(int argc, char** argv){
	ifstream fin;
	fin.open("inp.txt");

	ofstream fout("ted_cruz.jpg1.txt");
	ifstream fin1("ted_cruz.jpg.txt");

	int dig;
	int shx,shy; 

	int p1,p2;
	int choice=1;
		Mat t=Mat::ones(3,1,CV_32F);
		t.at<float>(0,0)=160;
		t.at<float>(1,0)=210;

		double angle;
	while(choice){
		cout<<"Enter your choice for grayscale images:-\n";
		cout<<"  1-image resizing \n";
		cout<<"  2-image rotation \n";
		cout<<"  3-image translation \n";
		cout<<"  4-image shear \n";
		cout<<"  5-image transformation affine \n";
		cout<<"  6-image negative \n";
		cout<<"  7-image log transformation \n";
		cout<<"  6-image gamme transformation \n";
		cout<<"  8-image piecewise_linear_tranformation \n";
		cout<<"  9-image Bit plane slifing \n";
		cout<<"  10-image reconstruction \n";
		cout<<"  11-image histogram equalization \n";
		cout<<"  12-image adpaptive histogram equalization \n";
		cout<<"  13-image histogram matching \n";  
		cout<<"  0 - to quit \n";  
		fin>>choice;
	 	Mat g_image1,res,res1,B;
		string path, path1;
		Mat shr=Mat::ones(2,3,CV_32F);
		Mat image,rot_mat,n,n1,m;
	 	Mat g_image,er;
		Point center = Point(0,0);
	

		float x,y;
		int opt;
		String windowName1 = "opencv g_imaged inp  "; 
		namedWindow(windowName1); 
		
		String windowName_ = "opencv g_imaged Ref  "; 
		namedWindow(windowName_); 

		String windowName1_ = "opencv g_imaged opencv "; 
		namedWindow(windowName1_); 

		String windowName2 = "opencv g_imaged error  "; 
		namedWindow(windowName2); 

		String windowName = "opencv g_imaged trans  "; 
		namedWindow(windowName); 
		Size dsize(0,0);

		switch(choice){
			case 1: cout<<"Resize >> ";
					cout<<"Image path - horizontal scaling(flaot) - vertical scaling(float) - interpolation_type(1 for nearest 0 for bilinear) \n";
					fin>>path>>x>>y>>opt;
					 // Read the g_image file
	 				image = imread(path);
					while(!fin1.eof()){
						fin1>>p1>>p2;
						fout<<round(x*p1)<<" "<<round(x*p2)<<endl;
					}
						 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=m_resize(g_image,x,y,opt);

					imshow(windowName1, g_image);
					imshow(windowName, res);
					resize(g_image,res1,dsize,y,x,INTER_LINEAR);
					 imshow(windowName1_, res1); 

					absdiff(res,res1,er);
					imshow(windowName2, er);
					cout<<rmse(res,res1)<<endl;
					 waitKey(0);

				 	break;
			case 2:  cout<<"Rotate >> ";
					cout<<"Image path - rotation-angle(in degrees) - interpolation_type(1 for nearest 0 for bilinear) \n";
					fin>>path>>x>>opt;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=rotate(x,g_image,opt);
					res1=Mat::zeros(g_image.rows,g_image.cols,g_image.type());

					center = Point(g_image.cols/2,g_image.rows/2);
					rot_mat = getRotationMatrix2D( center,x,1 );
					m=Mat::zeros(3,3,CV_32F);
					angle=x/180*M_PI;
					m.at<float>(0,0)=cos(angle);
					m.at<float>(0,1)=sin(angle);
					m.at<float>(1,0)=-sin(angle);
					m.at<float>(1,1)=cos(angle);
					m.at<float>(2,2)=1;
					n=m*t;
					dig=sqrt(g_image.rows*g_image.rows+g_image.cols*g_image.cols);

					
					shx=dig+5-g_image.rows;
					shy=dig+5-g_image.cols;

					
					//cout<<n.at<float>(0,0)-shx/2<<"  "<<n.at<float>(1,0)-shy/2<<" ";
					cout<<t.at<float>(0,0)<<"  "<<t.at<float>(1,0)<<" \n";
					t.at<float>(0,0)+=100;
					t.at<float>(1,0)+=100;

					warpAffine( g_image,res1, rot_mat, g_image.size() );
//					 imshow(windowName1_, res1); 

					imshow(windowName1, g_image);
					imshow(windowName, res );
					imwrite( "def.jpg", res );
					 waitKey(0);

				 break;

			case 3: cout<<"Translate >> ";
					cout<<"Image path - horizontal displace(+int) - vertical displace(+int) \n";
					int dx,dy;
					fin>>path>>dx>>dy;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=translate(dx,dy,g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);


			break;
			case 4: cout<<"Shear >> ";
					 cout<<"Image path - horizontal shear(int) - vertical shear(int) - interpolation_type(1 for nearest 0 for bilinear) \n";
					int sx,sy;
					fin>>path>>sx>>sy;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=shear(sx,sy,g_image,opt);

					imshow(windowName1, g_image);
					imshow(windowName, res);
					shr.at<float>(0,1)=sy;
					shr.at<float>(0,2)=0;
					shr.at<float>(1,2)=0;
					shr.at<float>(1,0)=sx;
					res1=Mat::zeros(g_image.rows,g_image.cols,g_image.type());

					warpAffine(g_image ,res1, shr, g_image.size() );
//					 imshow(windowName1_, res1); 



					 waitKey(0);
	 break;
			case 5:  cout<<"Negative >> ";
					cout<<"Image path \n";
					fin>>path;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=negative(g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);
	 break; break;
			case 6:  cout<<"Log transform >> ";
					cout<<"Image path - constant factor(double) \n";
					fin>>path>>x;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=log_t(x,g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;

			case 7: cout<<"Gamma transform >> ";
					 cout<<"Image path - constant factor(double) - gamma(double) \n";
					fin>>path>>x>>y;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=gamma_t(y,x,g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;

			case 8:  cout<<"Piecewise_transfom >> ";
					cout<<"Image path - start-x - start-y - end-x  - end-y \n";
					int r1,r2,s2,s1;
					fin>>path>>r1>>s1>>r2>>s2;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=piecewise_t(r1,s1,r2,s2,g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;

			case 9:  cout<<"Bit Plane >> ";
					cout<<"Image path - Plane(int: 0-8 ) \n";
					fin>>path>>dx;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=bit_plane(dx,g_image);
					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;

			case 10:cout<<"Reconstruction >> ";
					  cout<<"Image path - Tie points x1-y1-x2-y2-x3-y3-x4-y4-x1_-y1_-x2_-y2_-x3_-y3_-x4_-y4_  \n";
					double x1,y1,x2,y2,x3,y3,x4,y4,x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_;
					fin>>path>>x1>>y1>>x2>>y2>>x3>>y3>>x4>>y4>>x1_>>y1_>>x2_>>y2_>>x3_>>y3_>>x4_>>y4_;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=tie_points(x1,y1,x2,y2,x3,y3,x4,y4,x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_,g_image);

					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;break;
			case 11:  cout<<"Hist equalization >> ";
					cout<<"Image path \n";
					fin>>path;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=hist_eqal(g_image);
					equalizeHist( g_image, res1 );
					//imshow(windowName1_, res1); 

					absdiff(res,res1,er);
					imshow(windowName2, er);
					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;break;

			case 12:  cout<<"Adap_hist_eqal>> ";
					cout<<"Image path - Window rows -Window cols \n";
					fin>>path>>dx>>dy;
					 // Read the g_image file
	 				image = imread(path);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					res=adap_hist_eqal(dx,dy,g_image);
					imshow(windowName1, g_image);
					imshow(windowName, res);

					 waitKey(0);break;break;

			case 13:cout<<"Hist Match >> ";
					 cout<<"Image path - Refrence Image path  \n";
					fin>>path>>path1;
					 // Read the g_image file
	 				image = imread(path);
	 				B=imread(path1);
					 // Check for failure
					 if (image.empty())
					 {
					  cout << "Could not open or find the image" << endl;
					  break;
					 }
					cvtColor( image, g_image, CV_BGR2GRAY );
					cvtColor( B, g_image1, CV_BGR2GRAY );
					res=hist_match(g_image,g_image1); 
					imshow(windowName1, g_image);
					imshow(windowName, res);
					imshow(windowName_, B);
					 waitKey(0);

					break;
			case 0:  break;
			default: cout<<"Invalid choice\n";
		}
	}

 return 0;
}