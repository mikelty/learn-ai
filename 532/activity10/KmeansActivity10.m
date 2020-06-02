%% K-means Activity 3.1
%% Prepare Workspace

close all
clear

load('Activity3.1.mat')

%% Represent A using 1 cluster

p = 1 % number of clusters
Y = A'; %put cols of A in rows of Y, since kmeans builds clusters with rows

[idxA1,CA1] = kmeans(Y,p);

% Approximate A as the product of cluster centers times selection matrix

TA1 = CA1'
VA1_T = zeros(p,size(A,2));

for i = 1:size(A,2)
    VA1_T(idxA1(i),i) = 1;
end

VA1_T

Arank1 = TA1*VA1_T


%% Represent A using 2 clusters

p = 2 % number of clusters
Y = A'; %put cols of A in rows of Y, since kmeans builds clusters with rows

[idxA2,CA2] = kmeans(Y,p);

% Approximate A as the product of cluster centers times selection matrix

TA2 = CA2'
VA2_T = zeros(p,size(A,2));

for i = 1:size(A,2)
    VA2_T(idxA2(i),i) = 1;
end

VA2_T

Arank2 = TA2*VA2_T


