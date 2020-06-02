%% K means and SVD for Movie Ratings
%% Prepare Workspace

close all
clear

load('Period11Activity.mat')

disp('Ratings Matrix')
X

%% Two Clusters for Movie Ratings
%
% Using 2 clusters approximates each person's rating by one of two possible
% vectors - one which prefers sci-fi over romance, and another that
% prefers romance over sci-fi.  Neither of these cluster centers correspond
% exactly to anyone's individual ratings.

p=2; %number of clusters
Y = X'; %put cols of X in rows of Y, since kmeans builds clusters with rows

[idxX2,CX2] = kmeans(Y,p); % Use two clusters

TX2 = CX2'
VX2_T = zeros(p,size(X,2));

for i = 1:size(X,2)
    VX2_T(idxX2(i),i) = 1;
end

VX2_T

Xrank2 = TX2*VX2_T


%% Three Clusters for Movie Ratings
%
% Using 3 clusters approximates each person's rating by one of three
% possible vectors - one which slightly prefers sci-fi over romance, and
% another that prefers romance over sci-fi with lower absolute ratings, and
% a third that prefers romance over sci-fi with higher average ratings.
% None of these cluster centers correspond exactly to anyone's
% individual ratings.

p=3; %number of clusters
Y = X'; %put cols of X in rows of Y, since kmeans builds clusters with rows

[idxX3,CX3] = kmeans(Y,p); % Use two clusters

TX3 = CX3'
VX3_T = zeros(p,size(X,2));

for i = 1:size(X,2)
    VX3_T(idxX3(i),i) = 1;
end

VX3_T

Xrank3 = TX3*VX3_T

%% SVD Tastes
%
% Part a) T = U(:,1:r), W = S(1:r,1:r)*V(:,1:r)'
%
% That is, the tastes T are the first r columns of U, the r singular values
% corresponding to the largest r left singular vectors
%
% The weight matrix W consists of the first r rows of the product (S*V')

%% SVD Rank-1 Approx
%
% The signs of all entries of T are the same and they are approximately the
% same value, so this is capturing the (approximate) average rating of all
% movies by each user - you can see Jackson's ratings are much higher than
% Jasmine's.

[U,S,V] = svd(X,'econ');

disp('T for rank-1 approx')
U(:,1)
disp('W for rank-1 approx')
S(1,1)*V(:,1)'
disp('TW for rank-1 approx')
U(:,1)*S(1,1)*V(:,1)'

%% SVD Rank-2 Approx
%
% The sign of the entries in the second column of T are different for the
% romance movies than the sci-fi movies.  Hence this column is capturing
% differences in the two classes of movies.

disp('T for rank-2 approx')
U(:,1:2)
disp('W for rank-2 approx')
S(1:2,1:2)*V(:,1:2)'
disp('TW for rank-2 approx')
U(:,1:2)*S(1:2,1:2)*V(:,1:2)'

%% SVD Predict Jon's ratings
%
% Let T = U(:,1:2) be the first two tastes.  Then our prediction is Jon =
% T*a where a has two elements that weight the first and second tastes. We
% can use the ratings for the two movies we know to find a as follows:

T = U(:,1:2);
G = T(1:2,1:2); % tastes coeffs for first two movies

% now solve min_a || [6; 4] - G*a||^2_2

y = [6; 4];
a = inv(G'*G)*G'*y

% now use a to predict Jon's ratings for all movies

Jon_ratings = T*a

