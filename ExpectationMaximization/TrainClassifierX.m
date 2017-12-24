
function parameters = TrainClassifierX (train_input_data, train_input_labels) 

%% Written by Peter Sarvari, 2017
% Imperial College, London, ID: 00987075

% This is needed because in the code I assume that class labels start from
% one
if min(train_input_labels) == 0
    train_input_labels = train_input_labels + 1;
    disp('Samples have been relabelled so that the labels start from 1! Press enter to acknowledge!')
    pause;
end

%close all

load Activities.mat
maxiterkmeans = 100; %maximum iterations allowed in the kmeans algorithm
train_data_class = cell(1,length(unique(train_input_labels))); 
%length(unique(train_input_labels)) is the number of classes

for class = 1:length(unique(train_input_labels)) 
    %stores train data separately for each class
    train_data_class{class} = train_input_data(train_input_labels==class,:);
end



%% Kmeans with Elbow Method - Pipeline Step 1

% totalsumofsquares = zeros(length(unique(train_input_labels)),9);
% %stores the sum of (squared) distances of the data points 
% %from their assigned cluster centres for each class and each possible
% %maximum cluster option (2-10 clusters, see below)
% for class = 1:length(unique(train_input_labels)) %do for all classes
%     for init = 1:3 %see the result for 3 different initializations
%         for clusters = 2:10 %I investigate this region only
%             %choose the initial centroids randomly from data points
%             [centroids, ~] = kMeansInitCentroids(train_data_class{class}, clusters);
%             [classes, mu, totalsumofsquares(class,clusters-1)] = fastkmeans(train_data_class{class}, centroids, maxiterkmeans);
%             %totalsumofsquares
%             subplot(length(unique(train_input_labels))/2,2,class)
%         end
%         plot(2:10, totalsumofsquares(class,:), 'LineWidth',4);
%         hold on
%         ylabel('Sum of distances from assigned centroids', 'FontSize', 16);
%         xlabel('Number of clusters', 'FontSize', 16);
%         title(['Finding optimal number of clusters for class ', num2str(class)], 'FontSize', 20);
%         legend({'Initialization 1', 'Initialization 2', 'Initialization 3'}, 'FontSize', 16);
%     end
% end
% 
% pause;

%% Minimizing squared distance given a number of clusters - Pipeline Step 2
% From elbow method we chose 4, 3, 3, 4 for class 1, 2, 3, 4, respectively
% These were determined from Pipeline Step 1 (now commented out)

totalsumofsquares = zeros(length(unique(train_input_labels)),10);
% storing the totalsumofsquares (defined in Step 1) for each class and 
% each of the 10 random initialization
mu = cell(length(unique(train_input_labels)),10);
% cluster centres for each class and for each initialization
mu_best = cell(length(unique(train_input_labels)),1);
% stores the cluster centres for each class that resulted in the smallest
% totalsumofsquares among the 10 initializations (multiple initialization
% is implemented to avoid local minima)

for class = 1:length(unique(train_input_labels))
    for init = 1:10
        
        %Need to rewrite if new dataset accoding to results from Part I!!
        
        if class == 1 || class == 4
            clusters = 4; %4 for 2-class, 7 for binary
        else
            clusters = 3; %3 for 4-class, 6 for binary
        end
        
        %End of need to rewrite if new dataset
        
        %choose the initial centroids randomly from data points
        [centroids, ~] = kMeansInitCentroids(train_data_class{class}, clusters);
        %implement kmeans using function defined below
        [~, mu{class,init}, totalsumofsquares(class,init)] = fastkmeans(train_data_class{class}, centroids, maxiterkmeans);
        [~, ix] = min(totalsumofsquares, [], 2);
        mu_best{class} = mu{class, ix};
    end
end

%% EM - Pipeline Step 3

%Parameters of the EM algorithm: mean, covariance matrix and mixing 
%coefficients
mus = cell(1,length(unique(train_input_labels)));
covariances = cell(1,length(unique(train_input_labels)));
coeffs = cell(1,length(unique(train_input_labels)));

for class = 1:length(unique(train_input_labels))

    % Iinitializing EM parameters for each class separately
    covars = zeros(size(mus{class},1),size(mus{class},1),size(mus{class},2));
    % Centroids are initialized from Kmeans result above
    mus{class} = mu_best{class};
    for n = 1:size(mus{class},1) 
        %size(mus{class},1) is the number of clusters
        covars(:,:,n) = eye(size(mus{class},2)); 
        %size(mus{class},2) is the number of dimensions
    end
    coeffs{class} = 1/size(mus{class},1) * ones(size(mus{class},1), 1);
    %Coefficients are initialized so that they are uniform for the
    %Gaussians and they are also normalized
    covariances{class} = covars;

    disp('initial log likelihood is:');
    maxiter = 1000; %Maximum iteration for the EM algorithm
    p = zeros(maxiter+1, 1);
    %Calculating the log likelihood (see Bishop)
    p(1) = logLikelihoodGaussianMixture(coeffs{class}, (mus{class})', covariances{class}, train_data_class{class});
    p(1) %displaying initial log likelihood

    for iter=1:maxiter
        %Calculating the responsibilities of each Gaussian for the data
        %points (see Bishop). Note that mus matrix had to be transposed
        %because of the specification of the responsibilities function
        %E-step:
        gamma = responsibilities(coeffs{class}, (mus{class})', covariances{class}, train_data_class{class});
        %M-step
        combined_params = MaximizeProbability(train_data_class{class}, gamma);
        %Unwrapping the parameters
        coeffs{class} = combined_params{1};
        mus{class} = (combined_params{2})';
        covariances{class} = combined_params{3};
        %Showing number of iterations
        iter
        %Calculating and displaying new log likelihood to see how algorithm
        %converges
        p(iter+1) = logLikelihoodGaussianMixture(coeffs{class}, (mus{class})', covariances{class}, train_data_class{class});
        p(iter+1)
        if p(iter+1) < (p(iter) + 1) %stopping criterion, can be changed
            %but small enough compared to initial p (10^4)
            break
        end
    end
end

%% Priors - Part of Naive Bayes classification, Pipeline Step 4

%Wrapping parameters into the cell array called "parameters"
parameters = cell(1,4);
parameters{1} = coeffs;
parameters{2} = mus;
parameters{3} = covariances;
%Calculating prior probability of classes based on the training labels
prior = zeros(1,length(unique(train_input_labels)));
for class = 1:length(unique(train_input_labels))
    prior(class) = sum(train_input_labels==class)/length(train_input_labels);
end
parameters{4} = prior;

end

function [class_vec, mean_vec, totalsumofsquares] = fastkmeans(vec, initial_means, maxiterkmeans)
%Kmeans algorithm implementation
%Cluster 1 is first row vector in initial_means
mean_vec = initial_means;
mean_minus = zeros(size(initial_means));
index = 0;
K = size(initial_means, 1);
while ~isequal(mean_minus, mean_vec);
    mean_minus = mean_vec;
    res_matrix = bsxfun(@plus, sum(vec.^2, 2), bsxfun(@minus, (sum(mean_vec.^2, 2))', 2*vec*mean_vec')); 
    %Vectorization: I want (SAMPLEi - MUx)^2, 
    %I calculate SAMPLEi^2+MUx^2-SAMPLEi*MUx
    [sumofsquares, ix] = min(res_matrix, [], 2);
    class_vec = ix; %assign the data points to the cluster centre closest 
    %to them
    totalsumofsquares = sum(sumofsquares);
    for class = 1:K
        mean_vec(class,:) = mean(vec(class_vec==class, :), 1);
    end
    index = index + 1
    if index > maxiterkmeans
        break
    end
    %Note for me:
    %NaN was caused by centroids being the same
    %because multiple same entries in original dataset
    %pause
end

end

function [centroids, randidx] = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%idea from Prof. Andrew Ng's Coursera course
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X

% Initialize the centroids to be random examples
% Randomly reorder the indices of examples 
randidx = randperm(size(X, 1)); 
% Take the first K examples as centroids 
centroids = X(randidx(1:K), :);

end

function p = logLikelihoodGaussianMixture(coeffs, mus, covars, x)
%mus are the means of the K multivariate Gaussians (D*K), where D is the
%Diension and K is the number of Gaussians
%coeffs are the mixing coefficients, size is K*1
%covars are the covariance matrices of the K multivariate Gaussians (D*D*K)
%x are the samples matrix (M*D), where M is the number is samples
%see Bishop on the calculation of the log likelihood

p = 0;
tempo = 0;

for sample = 1:size(x, 1)
    for cluster_number = 1:size(mus, 2)
        %size(x(sample,:))
        %size(mus(:,cluster_number))
        %size(mus, 2)
        %size(coeffs(cluster_number))
        tempo = tempo + coeffs(cluster_number)*multivariateGaussian(x(sample,:), mus(:,cluster_number), covars(:,:,cluster_number));
    end
    p = p + log(tempo);
    tempo = 0;
end

end

function gamma = responsibilities(coeffs, mus, covars, x)
%coeffs are the mixing coefficients
%mus are the means of the k multivariate Gaussians (D*K), where D is the
%Diension and K is the number of Gaussians
%covars are the covariance matrices of the K multivariate Gaussians (D*D*K)
%x are the samples matrix (M*D), where M is the number is samples
%gamma has size M*K and stores the posteriors (responsibilities) for a
%sample and a particular Gaussian

%See Page 438, Bishop, Eq (9.23)


temp = zeros(size(x,1), size(mus, 2));
for cluster_number = 1:size(mus, 2)
    temp(:, cluster_number) = coeffs(cluster_number)*multivariateGaussian(x, mus(:, cluster_number), covars(:,:,cluster_number));
end

denominator = sum(temp, 2);

gamma = zeros(size(x,1), size(mus,2));

for sample = 1:size(x,1)
    for cluster_number = 1:size(mus,2)
        gamma(sample, cluster_number) = temp(sample, cluster_number)/denominator(sample);
    end
end

end

function combined_params = MaximizeProbability(x, gamma)
%Algorithm outlined in Bishop, page 439

mus = zeros(size(x,2), size(gamma, 2));
coeffs = zeros(size(gamma,2), 1);
covars = zeros(size(x,2), size(x,2), size(gamma,2));
for cluster_number = 1:size(gamma,2)
    Nk = sum(gamma(:,cluster_number)); %Eq (9.27)
    %Eq (9.24)
    mus(:,cluster_number) = (1/Nk * sum(bsxfun(@times, x, gamma(:,cluster_number)),1))';
    %Eq (9.26)
    coeffs(cluster_number) = Nk/size(x,1);
    %Eq (9.25)
    temp = zeros(size(x,2), size(x,2));
    for sample = 1:size(x,1)
        temp = temp + gamma(sample, cluster_number)*((x(sample,:))'-mus(:,cluster_number))*(x(sample,:)-(mus(:,cluster_number))');
    end
    covars(:,:,cluster_number) = 1/Nk * temp;
end
%Wrap parameters into cell array called "combined_params"
combined_params = cell(3,1);
combined_params{1} = coeffs;
combined_params{2} = mus;
combined_params{3} = covars;
end

function p = multivariateGaussian(X, mu, Sigma2)
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%    X should have dimension n by k, where n is the number of samples and k
%    is the number of features (base functions)
%    Normalized distribution and vectorized implementation

k = length(mu); %Dimension of the Gaussian

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)'); %(:) makes mu a column vector so it does not matter if input is row or column vector
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end