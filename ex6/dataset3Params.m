function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
    CVect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    sigmaVect = CVect;

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
    
    paramMat = nan(length(CVect) * length(CVect), 3);
    idx = 1;
    
    for c = 1:length(CVect)
        for s = 1:length(sigmaVect)
            model = svmTrain(X, y, CVect(c), @(x1, x2) gaussianKernel(x1, x2, sigmaVect(s)));
            pred = svmPredict(model, Xval);
            paramMat(idx, :) = [CVect(c) sigmaVect(s) mean(double(pred ~= yval))];
            idx = idx + 1;
        end
    end
    
    [~, maxIdx] = min(paramMat(:, 3));
    C = paramMat(maxIdx, 1);
    sigma = paramMat(maxIdx, 2);
% =========================================================================

end
