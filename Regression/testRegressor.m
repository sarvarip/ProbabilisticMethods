function y_pred = testRegressor(X, param)
%Written by Peter Sarvari, 00987075

%%  ======= Part 1: Retrieve the base functions and the calculated weights from training ======== 

baseFuncs=param.baseFuncs;
w=param.w;
no_bases = length(baseFuncs);

%%  ======= Part 2: Creating the teta design matrix for test samples ======== 

teta = zeros(size(X, 1), no_bases+1);
teta(:,end) = ones(size(X,1),1); %bias
for base = 1:no_bases           
    teta(:,base) = baseFuncs{base}(X);
end

%%  ======= Part 3: Predicting the price for test samples ======== 

y_pred = teta*w;





