function [ labels, bestg, bestc, bestcv ] = do_svm_on_hog( hog_train, hog_test, y_train )


data = scaleData([hog_train;hog_test]);
hog_train = data(1:size(hog_train,1),:);
hog_test = data(size(hog_train,1)+1:end,:);
bestcv = 0;
iter = 1;
for log2c = 0:6,
  for log2g = -6:-2,
    cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(double(y_train), double(hog_train), cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    cmd1 = ['-c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    model = svmtrain(double(y_train), double(hog_train), cmd1);
    [labels, acc, dec] = svmpredict(ones(size(hog_test,1),1),double(hog_test), model);
    assert(size(labels,1)==size(hog_test,1));
    k=sprintf('labels_new%d.csv',iter)
    writeLabels(k, labels);
    iter = iter+1;
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end
end

