function transformedImage = transformImageBackward(img,A)
    A = [A,zeros(2,1);zeros(1,2),1];
    iA = inv(A);
    tform = affine2d(iA');
    transformedImage = imwarp(img,tform);
    %imshow(transformedImage)    
end