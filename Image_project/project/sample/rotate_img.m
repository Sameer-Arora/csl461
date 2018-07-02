img1 = imread('cameraman.tif');

img2 = imrotate(img1,30,'bilinear','crop');
%img3 = imrotate(img1,30,'bilinear');
imshow(uint8(img2));
%imshow(img2,'InitialMagnification','fit');
imwrite(img2,'sampleimg1.tif');
%
%imshowpair(img1,img2,'montage');
%img3 = imread('sampleimg1.tif');
imshow(img2);
%imshow(img3);