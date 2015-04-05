function [ ypred ] = ensemble( X_train,y_train,X_test)
    model = fitensemble(X_train,y_train,'bag',10000,'tree','type','Classification');
    ypred = predict(model,X_test);
end

