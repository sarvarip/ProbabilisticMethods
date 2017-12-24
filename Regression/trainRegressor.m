function [ param ] = trainRegressor(X, y)
%Witten by Peter Sarvari, 00987075

%%  ======= Part 1: Choosing mus and number of basis functions ======== 

%load('metro.mat') 
%Mus of selected tube stations - for the selection process see Appendix

   metro = [51.5025   -0.2781
   51.5088   -0.2630
   51.5139   -0.0754
   51.5151   -0.0718
   51.5410   -0.3006
   51.5325   -0.1058
   51.5654   -0.1347
   51.6163   -0.1336
   51.5585   -0.1056
   51.5226   -0.1570
   51.4434   -0.1529
   51.5209   -0.0978
   51.5408    0.0802
   51.4903   -0.2143
   51.5122   -0.1876
   51.5403    0.1262
   51.5504   -0.1646
   51.4981   -0.0635
   51.5272   -0.0550
   51.5870   -0.0410
   51.5010   -0.0945
   51.4954   -0.3256
   51.6070   -0.1242
   51.5271   -0.0248
   51.5769   -0.2141
   51.4624   -0.1154
   51.5247   -0.0128
   51.5451   -0.2030
   51.5406   -0.2109
   51.6030   -0.2643
   51.5485   -0.1180
   51.5433   -0.1154
   51.5419   -0.1398
   51.5393   -0.1432
   51.4979   -0.0497
   51.5036   -0.0199
   51.5482   -0.0963
   51.5439   -0.1538
   51.5184   -0.1112
   51.4947   -0.2687
   51.4616   -0.1380
   51.4652   -0.1295
   51.4529   -0.1476
   51.5956   -0.2500
   51.4182   -0.1780
   51.5483   -0.0788
   51.5525   -0.2388
   51.5148   -0.3012
   51.5100   -0.2883
   51.4915   -0.1939
   51.5175   -0.2483
   51.5873   -0.1650
   51.5393    0.0526
   51.4590   -0.2111
   51.6138   -0.2752
   51.5196   -0.1691
   51.5202   -0.1670
   51.4947   -0.1005
   51.5277   -0.1330
   51.5258   -0.1357
   51.5204   -0.1051
   51.6010   -0.1924
   51.5467   -0.1799
   51.5499   -0.1841
   51.5650   -0.1054
   51.4803   -0.1949
   51.4943   -0.1828
   51.5725   -0.1941
   51.5012   -0.2268
   51.5206   -0.1344
   51.5560   -0.1523
   51.5239   -0.1440
   51.5423   -0.3459
   51.4913   -0.2756
   51.5469   -0.0578
   51.4923   -0.2228
   51.4942   -0.2260
   51.5564   -0.1785
   51.5553   -0.1666
   51.5300   -0.2929
   51.5363   -0.2576
   51.5921   -0.3347
   51.5793   -0.3371
   51.4670   -0.4230
   51.4713   -0.4527
   51.5835   -0.2265
   51.5003   -0.1924
   51.5450   -0.1075
   51.5772   -0.1455
   51.5536   -0.4506
   51.5171   -0.1206
   51.5529   -0.1128
   51.5471   -0.0432
   51.4713   -0.3659
   51.4735   -0.3560
   51.4737   -0.3865
   51.4882   -0.1059
   51.5306   -0.2243
   51.4967   -0.2093
   51.5505   -0.1407
   51.5478   -0.1471
   51.5817   -0.3169
   51.4773   -0.2850
   51.5472   -0.2050
   51.5350   -0.1940
   51.5306   -0.1240
   51.5308   -0.1204
   51.5173   -0.2111
   51.4989   -0.1122
   51.5121   -0.1751
   51.5131   -0.2184
   51.5115   -0.1285
   51.5565   -0.0057
   51.5683    0.0074
   51.5177   -0.0825
   51.5054   -0.0848
   51.5299   -0.1859
   51.5704   -0.0960
   51.5136   -0.1587
   51.5227   -0.1630
   51.5253   -0.0335
   51.4022   -0.1946
   51.5343   -0.1390
   51.5543   -0.2504
   51.4774   -0.0339
   51.4759   -0.0405
   51.5237   -0.2602
   51.5006    0.0036
   51.5626   -0.3041
   51.4999   -0.3132
   51.5481   -0.3683
   51.6110   -0.4236
   51.6004   -0.4088
   51.5090   -0.1963
   51.6474   -0.1319
   51.5262   -0.0875
   51.4816   -0.3520
   51.4821   -0.1128
   51.5152   -0.1412
   51.5152   -0.1755
   51.5191   -0.1770
   51.4749   -0.2021
   51.5367   -0.3239
   51.5102   -0.1339
   51.4891   -0.1330
   51.5927   -0.3805
   51.5312    0.0166
   51.5723   -0.2958
   51.4687   -0.2088
   51.5342   -0.2053
   51.4942   -0.2356
   51.5753   -0.3724
   51.4634   -0.3020
   51.5009   -0.0521
   51.5194   -0.1878
   51.5232   -0.1243
   51.4997   -0.1339
   51.5832   -0.0750
   51.5112   -0.0569
   51.5047   -0.2188
   51.5061   -0.2262
   51.4926   -0.1561
   51.5800    0.0213
   51.4996   -0.2702
   51.5008   -0.3079
   51.5646   -0.3521
   51.4941   -0.1728
   51.5568   -0.3987
   51.4154   -0.1923
   51.5909    0.0273
   51.4449   -0.2065
   51.6324   -0.1277
   51.5038   -0.1048
   51.5347   -0.1743
   51.4949   -0.2462
   51.5219   -0.0465
   51.4722   -0.1227
   51.5412   -0.0038
   51.5570   -0.3359
   51.5509   -0.3160
   51.4937   -0.0482
   51.5433   -0.1744
   51.4358   -0.1596
   51.4278   -0.1679
   51.5164   -0.1303
   51.5889   -0.0598
   51.6302   -0.1792
   51.5099   -0.0768
   51.5566   -0.1381
   51.4951   -0.2541
   51.5903   -0.1031
   51.5382    0.1000
   51.5351    0.0337
   51.5469   -0.4773
   51.4860   -0.1237
   51.4966   -0.1440
   51.5829   -0.0200
   51.5757    0.0286
   51.5045   -0.0561
   51.5245   -0.1381
   51.5233   -0.1838
   51.5512   -0.2958
   51.5636   -0.2798
   51.5182   -0.2806
   51.4868   -0.1948
   51.6092   -0.1887
   51.5279    0.0041
   51.5468   -0.1910
   51.5795   -0.3531
   51.4909   -0.2053
   51.5210   -0.2018
   51.5195   -0.0595
   51.5492   -0.2211
   51.4220   -0.2054
   51.4339   -0.1986
   51.5971   -0.1094
   51.6058    0.0333
   51.6182   -0.1856];

mu = metro;
no_bases = size(mu,1);

%%  ======= Part 2: Creating the teta design matrix ========

param.mu = cell(no_bases,1);
param.sigma = cell(no_bases,1);
param.baseFuncs = cell(no_bases,1);

teta = zeros(size(X, 1), no_bases+1);
teta(:,end) = ones(size(X,1),1); %bias weight
X_mean = mean(X,1); %Coordinate means
covar = 0.1 * (1/size(X,1)) * (bsxfun(@minus, X', X_mean')*bsxfun(@minus, X, X_mean)); %Covariance matrix scaled by 0.1 (see main text for reason)
if nargin == 4 %only when the script is used for hyperparameter tuning
    covar = covariance_scale*(1/size(X,1)) * (bsxfun(@minus, X', X_mean')*bsxfun(@minus, X, X_mean)); 
end
for base = 1:no_bases          
    param.baseFuncs{base}=@(X_in)multivariateGaussian(X_in,mu(base,:),covar);
    teta(:,base) = param.baseFuncs{base}(X);
    param.mu{base}=mu(base,:);
    param.sig{base}=covar;
end

%%  ======= Part 3: Calculating the weights matrix ========

if nargin == 4 
    if lambda ~= 0
        param.w = (lambda*eye(no_bases+1)+teta'*teta)\teta'*y; %Equation (3.28), Bishop
    else
        param.w = pinv(teta)*y; %pseudoinverse for least squares solution
    end
else 
    param.w = pinv(teta)*y; %pseudoinverse for least squares solution
end

param.teta = teta;

end

%%  ======= Part 4: Calculating base function values ========

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

% k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)'); %(:) makes mu a column vector so it does not matter if input is row or column vector

%Use this to get normalized probability distribution:

% p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
%     exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

%Use this for not normalized probability distribution:

p = exp(-0.5 * sum(X*inv(Sigma2).*X,2));


end