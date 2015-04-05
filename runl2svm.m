function runl2svm(dataset, opt)
%Run this with project starter handout
%RUNCLASSIFIER Runs a simple SVM or MLR classifier.
% dataset - either 'random' or './path/to/dataset/' containing
%           entries X_train, X_test, y_train, (y_test - optional).
% opt     - options to run with:
%     .loss     - 'mlr' for softmax regression and 'l2svm' for L2 SVM.
%                 Default is 'mlr'.
%
%     .lambda   - regularization parameter. Default is 0.
%
%     .dual     - optimize in the dual if true. Default is false. If false
%                 then a linear kernel is used.
%
%     .kernelfn - kernel function - Either a string 'rbf' for RBF kernel or
%                 'poly' for a polynomial kernel.
%                 Alternatively, kernelfn can be a function kernelfn(x, y)
%                 which should return an m1 x m2 gram matrix between 
%                 x and y, where there are m1 examples in x and m2 in y.
%                 For example you can implement a tanh kernel with params
%                 a and b as opt.kernelfn = @(X1, X2) tanh(a*X1*X2' - b).
%                 Default is 'rbf'.
%
%     .gamma    - RBF kernel width. Larger gamma => smaller variance.
%                 gaussian. Default is 1.
%
%     .order    - Polynomial order. Default is 3.
%   
    
            if nargin < 2
                % parameters you can play with.
                %lambda was 1
                %PCA 150 dimension turns out to be better tried till 200
                opt.lambda = 0.01;        % regularization
                opt.loss = 'l2svm';      % 'mlr' for Multinomial Logistic Regression
                % (softmax) or 'l2svm' for L2 SVM.
                opt.dual = true;      % optimize dual problem
                % (must be true to use kernels)
                opt.kernelfn = 'rbf';  % kernel to use (either rbf or poly)
                opt.gamma = 0.8;      % Kernel parameter for RBF kernel.
                opt.order = 2;         % Kernel parameter for polynomial kernel.
            end
            
            % type the following into the matlab terminal to compile minFunc:
            % >> addpath ./minFunc/
            % >> mexAll
            addpath(genpath('./minFunc/'));
            addpath ./tinyclassifier/
            addpath ./helpers
            
            % train and test classifier
            params = trainClassifier(X_train(1:end,:), y_train(1:end), opt);
            preds = predictClassifier(params, X_train);
            fprintf('Train Accuracy = %.2f%%\n', 100*mean(preds(:) == y_train(:)));
            preds = predictClassifier(params, X_test);
            fprintf('Test Accuracy = %.2f%%\n', 100*mean(preds(:) == y_test(:)));
            
            % write the data out to a file that can be read by Kaggle.
            k=sprintf('my_labels.csv');
            writeLabels(k, preds);
end
