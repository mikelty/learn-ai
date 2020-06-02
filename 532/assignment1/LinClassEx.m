%% Example linear classifier for randomly generated features


% number of features
p = 2;

% number of examples
n = 5000;

% generate matrix of n (random) examples of p features with a column of all ones
X = [rand(n,p), ones(n,1)];

% Classifier weights
w = [-1; 2; -0.4]

% Multiply feature matrix with weights
yhat = X*w;

% Decide which class based on whether yhat is >< 0
% sign function returns +1 when yhat(i)>0 and -1 when yhat(i)<0
class = sign(yhat);

% Display results
figure(1);clf;
scatter(X(:,1),X(:,2),40,class,'filled');
colorbar
set(gca,'FontSize',20)
xlabel('feature 1')
ylabel('feature 2')
title('Case 2')
