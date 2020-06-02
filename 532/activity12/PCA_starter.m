%% Activity 12 One-dimensional PCA
%% Prepare workspace

close all
clear

load('PCA_Activity.mat')

%% Display data

% Use rotate tool in the figure to view data from different angles
figure
scatter3( X(1,:), X(2,:), X(3,:), 'r.', 'LineWidth', 3 )
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

%% Subtract Mean

Y = X;
Y = X - mean(X,2)*ones(1,size(X,2));

%% Take SVD to find best line

[U,S,V] = svd(Y,'econ');

a =  ;  % Complete this line

%% Display best line on scatterplot

figure
scatter3( Y(1,:), Y(2,:), Y(3,:), 'r.', 'LineWidth', 3 )
xlabel('y_1')
ylabel('y_2')
zlabel('y_3')
hold on
plot3([0;a(1)],[0;a(2)],[0;a(3)], 'b', 'LineWidth', 2)