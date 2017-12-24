%% ======= Part 1: Clear workspace, load data ========

clc, clear
close all

load('london.mat')

%% ======= Part 2: Data cleaning ========
X = Prices.location;
target = Prices.rent;

%Assumption on the location of the useful data
target = target(X(:,1)<52); %IT IS VERY VERY IMPORTANT TO DO THIS FIRST BECAUSE IF I OVERWRITE X, I WILL ALWAYS DELETE THE LAST SAMPLES OF TARGET
X = X(X(:,1)<52,:);
target = target(X(:,1)>51);
X = X(X(:,1)>51,:);
target = target(X(:,2)>-0.5);
X = X(X(:,2)>-0.5,:);
target = target(X(:,2)<0.4);
X = X(X(:,2)<0.4,:);

%Getting rid of multiple entries
out = unique([X, target],'rows');
X = out(:,1:2);
target = out(:,3);

%Getting rid of conflicting entries
if ~isequal(X,unique(X, 'rows'))
    [X,I,~] = unique(X, 'rows', 'first'); %could have taken the average of conflicting prices of samples, but we have enough (N>>D)
    target = target(I);
end

%Assumption on the prices of the useful data
X = X(target<2000,:);
target = target(target<2000);
X = X(target>300,:);
target = target(target>300);
y = target;


%Simple anomaly detection to further filter out the outliers
X_mean = mean(X,1);
covar = 1/(size(X,1))*(bsxfun(@minus, X', X_mean')*bsxfun(@minus, X, X_mean));
probab = @(X_in)multivariateGaussian(X_in, X_mean, covar);
prob_vals = probab(X);
y = y(prob_vals>3*10^-2,:); %Arbitrary limit, 3 samples dropped
X = X(prob_vals>3*10^-2,:);

%Normalization - NOT USED
% avg = mean(X,1);
% standard = std(X,1);
% X = bsxfun(@minus, X, avg);
% X = bsxfun(@rdivide, X, standard);

%3D plotting of cleaned samples
scatter3(X(:,1), X(:,2), y, 'b');
hold on

%Tube location cleaning 
%Same assumption on location as before
metro = Tube.location;
metro = metro(metro(:,1)<52,:);
metro = metro(metro(:,1)>51,:);
metro = metro(metro(:,2)>-0.5,:);
metro = metro(metro(:,2)<0.4,:);

%Normalization of metro stations - NOT USED
% metro = bsxfun(@minus, metro, avg);
% metro = bsxfun(@rdivide, metro, standard);

%GET RID OF TUBE STATIONS WITHOUT HOUSES CLOSE
res_matrix = bsxfun(@plus, sum(X.^2, 2), bsxfun(@minus, (sum(metro.^2, 2))', 2*X*metro')); 
%I want (SAMPLEi - MUx)^2, I calculate SAMPLEi^2+MUx^2-2*SAMPLEi*MUx, which is the same but can be vectorized
%res_matrix is the distance how far sample i is from tube station x
[~, ix] = min(res_matrix, [], 2); %which tube station is closest to each sample
threshold = 5; %GET RID OF TUBE STATIONS IF ONLY 5 HOUSES OR LESS ARE THE CLOSEST TO IT - LESS CHANCE TO OVERFIT
centers = unique(ix);
metro_mus = [];
for i = 1:numel(centers)
    loc = centers(i);
    num = sum(ix==loc);
    if num > threshold
       metro_mus = [metro_mus loc];
    end
end
metro = metro(metro_mus,:);

%2D plotting of tube stations on the same graph
scatter3(metro(:,1), metro(:,2), zeros(size(metro,1),1), 'r');
title('Blue cicrles are houses and red circles are metro stations')
xlabel('Latitude')
ylabel('Longitude')
zlabel('Price')

%save('metro.mat', 'metro')

%% ======= Part 3: Use Cross Validation (K-folds) to choose best lambda ========
% My own method
% See main text for discussion of the method
lambda = [0]; %use to test trainRegressor
%lambda = [0, 10^-7 10^-6 10^-5]; %use to tune hyperparameters
covariance_scale = [0.1]; %use to test trainRegressor
%covariance_scale = [0.05 0.1 0.25]; %use to tune hyperparameters
[A,B] = meshgrid(lambda,covariance_scale);
c=cat(2,A',B');
options=reshape(c,[],2); %all possible combinations of lambda and covariance_scale
m = size(X,1);
seq = randperm(m); %Shuffle the data
series = 0.1:0.1:0.9; %K = 10 folds
mse_min = 10^6; %Set this big enough
test_mse = zeros(size(options, 2), (numel(series)+1)/2*(numel(series)+1)/2); %only works if K is even!
train_mse = zeros(size(test_mse));
crossval_mse = zeros(size(test_mse));
crossval_mse_mean = zeros(size(options));
X_K = cell((numel(series)+1), 1);
y_K = cell((numel(series)+1), 1);

%Partition the data into K folds

X_K{1} = X(seq(1:ceil(series(1)*m)), :);
y_K{1} = y(seq(1:ceil(series(1)*m)), :);
for i = 2:numel(series)
  idx = ceil(series(i-1)*m):ceil(series(i)*m);
  X_K{i} = X(seq(idx), :);
  y_K{i} = y(seq(idx), :);
end
X_K{numel(series)+1} = X(seq(ceil(series(end)*m):end), :);
y_K{numel(series)+1} = y(seq(ceil(series(end)*m):end), :);

%This is what the above code effectively does

%X_1 = X(seq(1:ceil(0.10*m)), :);
%X_2 = X(seq(ceil(0.10*m):ceil(0.20*m)), :);
%X_3 = X(seq(ceil(0.20*m):ceil(0.30*m)), :);
%X_4 = X(seq(ceil(0.30*m):ceil(0.40*m)), :);
%X_5 = X(seq(ceil(0.40*m):ceil(0.50*m)), :);
%X_6 = X(seq(ceil(0.50*m):ceil(0.60*m)), :);
%X_7 = X(seq(ceil(0.60*m):ceil(0.70*m)), :);
%X_8 = X(seq(ceil(0.70*m):ceil(0.80*m)), :);
%X_9 = X(seq(ceil(0.80*m):ceil(0.90*m)), :);
%X_10 = X(seq(ceil(0.90*m):end), :);

for i = 1:size(options,1)
  for j = 1:((numel(series)+1)/2) %crossval set index in the mse matrix, only works if K is even
    X_crossval = X_K{j};
    y_crossval = y_K{j};
    for k = ((numel(series)+1)/2)+1:(numel(series)+1) %test set index in the mse matrix 
        ind = (j-1)*(numel(series)+1)/2+k-((numel(series)+1)/2);
      if k~= j
        X_test = X_K{k};
        y_test = y_K{k};
        folds = 1:(numel(series)+1);
        X_train = vertcat(X_K{folds(folds~=j & folds~=k), :}); %training data
        %is the 80% that is not chosen for validation or testing
        y_train = vertcat(y_K{folds(folds~=j & folds~=k), :});
        
        %model = algo(X_train,y_train)
        %use this to tune hyperparameters:
        %param = trainRegressor_crossval(X_train, y_train, options(i,1), options(i,2));
        %use this to test trainRegressor:
        param = trainRegressor(X_train, y_train);
        
        %y_pred_train = model(X_train);
        y_pred_train = testRegressor(X_train, param);
        
        %y_pred_crossval = model(X_crossval);
        y_pred_crossval = testRegressor(X_crossval, param);
        
        %y_pred_test = model(X_test);
        y_pred_test = testRegressor(X_test, param); 
        
        train_mse(i,ind) = mse(y_train, y_pred_train); %put it in the relevant location in the mse matrices
        crossval_mse(i,ind) = mse(y_crossval, y_pred_crossval);
        test_mse(i,ind) = mse(y_test, y_pred_test);
        
      end
    end
  end
  
  %Choose the best hyperparameters - the combination that on average 
  %reduces the MSE in the cross-validation set the most
  crossval_mse_mean(i) = mean(crossval_mse(i, :));
  if crossval_mse_mean(i) < mse_min
    mse_min = crossval_mse_mean(i);
    best_lambda_mse = options(i,1);
    best_covariance_scale_mse = options(i,2);
    test_mse_best_option = mean(test_mse(i, :));
    train_mse_best_option = mean(train_mse(i,:));
    crossval_mse_best_option = mse_min;
  end
i %Reporting which combination of the options the algorithm is working on
end

%% ======= Part 4: Results ========

%Report the relevant results - see Table, main text
best_lambda_mse
best_covariance_scale_mse
test_mse_best_option
train_mse_best_option
crossval_mse_best_option

