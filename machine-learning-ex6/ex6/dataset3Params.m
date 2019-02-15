function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_all = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_all = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C_saved=0;
sigma_saved=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error_saved=9999;C_saved=0;sigma_saved=0;
for itr=1:8,
  for itr2=1:8,
  model= svmTrain(X, y, C_all(itr), @(x1, x2) gaussianKernel(x1, x2, sigma_all(itr2)));
  predictions = svmPredict(model, Xval);
  error=mean(double(predictions ~= yval));
  printf("error found is:%f for C=%f and sigma=%f /n",error,C_all(itr),sigma_all(itr2));
    if error < error_saved
     C_saved=C_all(itr);sigma_saved=sigma_all(itr2);
     printf("\nnew small error as error %f < error_saved %f\n",error,error_saved);
     error_saved=error;
    end
  end 
end
 

C=C_saved;
sigma=sigma_saved;



% =========================================================================

end
