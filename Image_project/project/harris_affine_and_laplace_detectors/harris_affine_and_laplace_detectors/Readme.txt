Steps:

1. For Harris-Affine Detector:
	a. img = imread(imagepath)
	b. affine_invariant_points = harris_affine(img)
2. For Harris-Laplace Detector:
	a. same as above
	b. scale_invariant_points = harris_laplace(img)
3. For Harris Interest Point Detector
	a. Provide the image path in the varaible "imagepath"


The other functions are called from harris_affine.m

Any change or correction is highly appreciated.
