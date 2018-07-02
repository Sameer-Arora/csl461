The code can be compiled using :
 g++ 1.cpp  `pkg-config --cflags opencv --libs opencv` 

 ./a.out 

The various techinques are used as given in the book.

For rotation,shear,tranlate the image is resized and thus we can't compare with opencv's results.

For resizing , histogram equlaization only the errors are computed.

The assignment is done indiviusually by myself, with help of class notes , opencv documentation and adaptive histogram from Wikipedia. 

For tie points its assumed that the tie pionts are given correctly.

The affine transformation can be applied by using the functions again and again.

The images are resized in the cases of rotation ,translation ,shear and tie points so no cutting of image happens as compared to inbuilt functions for them.

For tie points i assume that correct mapping is given for correct functioning of the functions.

For adaptive histogram equalization I have implemented both sliding window and tile adaptive histogram equalization.

For the tie points i have used a rough estimate of the original image size suach that image is not cropped at all.