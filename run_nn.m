function [preds] = run_nn(X_train,y_train, X_test)
% Trains a single layer NN
%Use it with homework3 code handout 
    addpath ./helpers
   
    n_classes = 10;
    %l=[6,7,8,9,10];
    opt.hidden_sizes = 256;
    opt.lambda = 0.05;
    opt.MaxIter = 1000; % max iterations for minimization function.
    opt.beta = 0.5;
    %beta was 0.5 giving 96%
    opt.p = 0.01;
    theta = nnTrainClassification(X_train, y_train, opt);
    trainpreds = nnPredictClassification(X_train, theta, n_classes, opt);
    accuracy=100*mean(trainpreds'==y_train)
    preds = nnPredictClassification(X_test, theta, n_classes, opt)';
    k=sprintf('Labels_new_%d.csv',opt.hidden_sizes);
    writeLabels(k,preds);
end