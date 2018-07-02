function transformedPoints = transformBackward(img_points,A)
    iA=inv(A);
    transformedPoints = [iA(1,1:2)*img_points'; iA(2,1:2)*img_points']';
end