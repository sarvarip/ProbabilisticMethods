function [pred_class, norm_results] = ClassifyX(input, parameters)

%% Written by Peter Sarvari, 2017
% Imperial College, London, ID: 00987075

%% Naive Bayes Classification - Pipeline Step 4

results = zeros(size(input,1), length(parameters{1})); %length(parameters{1}) is the number of classes

for classes = 1:length(parameters{1})
    temp = zeros(size(input,1), 1);
    %length(parameters{1}{classes}) is the number of clusters within each
    %class
    for clusters = 1:length(parameters{1}{classes}) 
        %Vectorized implementation of Eq (9.7) in Bishop, page 430
        temp = temp + parameters{1}{classes}(clusters) * multivariateGaussian(input, parameters{2}{classes}(clusters,:), parameters{3}{classes}(:,:,clusters)); %likelihood
    end
    results(:, classes) = temp * parameters{4}(classes); %posterior
end
norm_results = bsxfun(@rdivide, results, sum(results, 2)); %divide by p(x) 
%or in other words, normalize the result
[~, ix] = max(results, [], 2); %The predicted class is the one with the 
%highest posterior, p(class|x)
pred_class = ix;

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

