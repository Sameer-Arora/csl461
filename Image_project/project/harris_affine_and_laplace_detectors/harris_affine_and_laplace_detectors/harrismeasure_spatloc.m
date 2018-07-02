function R = harrismeasure_spatloc(W,sigmaI,sigmaD)
  %disp('harrismeasure')
  si = sigmaI;
  sd = sigmaD;
  imgxx = calcimgxx(W,si,sd);
  imgyy = calcimgyy(W,si,sd);
  imgxy = calcimgxy(W,si,sd);
  R = zeros(5,5);
  for i=1:5
    for j=1:5
      Mu = (sd^2)*[imgxx(i,j),imgxy(i,j);imgxy(i,j),imgyy(i,j)];
      [~,D] = eig(Mu);
      R(i,j) = det(D) - 0.05*(trace(D))^2;
    end
  end
end
