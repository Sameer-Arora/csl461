function transformedImage = transformImageForward(img,A)
    A = [A,zeros(2,1);zeros(1,2),1];
    tform = affine2d(A');
    transformedImage = imwarp(img,tform);
    %imshow(transformedImage)    
end