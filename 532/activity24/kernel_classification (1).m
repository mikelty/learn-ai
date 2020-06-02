%% Kernel classification - squared error

%% Set up workspace
close all
clear

p = 2;  % features
n = 1000;  % examples

sigma = 5;
lam = 0.01;

%% Generate training data

X = rand(n,p)-.5;  % n examples of p features
Y1 = 2*(sum(X.^2,2)>.1)-1;  % labels for first class problem
Y2 = 2*(5*X(:,1).^3 > X(:,2)) -1;  % labels for 2nd class problem
Y = [Y1 Y2];

%% Train classifier

distsq=zeros(n,n);
for i =1:n
    for j=1:n
        distsq(i,j) = (X(i,:)-X(j,:))*(X(i,:)-X(j,:))';
    end
end

K = exp(-distsq/(2*sigma^2));

alpha1 = inv(K+lam*eye(n))*Y1;
alpha2 = inv(K+lam*eye(n))*Y2;

%% Predict labels

%Yhat = K*Y;

Yhat = K*[alpha1,alpha2];

%% Display results

% First classification problem
figure
subplot(221);
scatter(X(:,1),X(:,2),20,Y1,'filled');
title('training data, label 1'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

subplot(222);scatter(X(:,1),X(:,2),20,Yhat(:,1),'filled');
title(['predicted label 1, \sigma = ', num2str(sigma,2)]); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

subplot(223);scatter(X(:,1),X(:,2),20,2*(Yhat(:,1)>0)-1,'filled');
title(['thresh label 1, \sigma = ', num2str(sigma,2)]); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

err = abs((2*(Yhat(:,1)>0)-1)-Y(:,1))/2;
subplot(224); scatter(X(:,1),X(:,2),20,(err>0),'filled');colormap cool
set(gca,'fontsize',14)
title([num2str(sum(err)),' Errors $\hat{y}_1 \neq y_1$'],'interpreter','latex')
axis image;colorbar; 

% Second classification problem
figure
subplot(221);
scatter(X(:,1),X(:,2),20,Y2,'filled');
title('training data, label 2'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

subplot(222);scatter(X(:,1),X(:,2),20,Yhat(:,2),'filled');
title(['predicted label 2, \sigma = ', num2str(sigma,2)]); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

subplot(223);scatter(X(:,1),X(:,2),20,2*(Yhat(:,2)>0)-1,'filled');
title(['thresh label 2, \sigma = ', num2str(sigma,2)]); 
axis image;colorbar;colormap jet; set(gca,'fontsize',14)

err = abs((2*(Yhat(:,2)>0)-1)-Y(:,2))/2;
subplot(224); scatter(X(:,1),X(:,2),20,(err>0),'filled');colormap cool
set(gca,'fontsize',14)
title([num2str(sum(err)),' Errors $\hat{y}_2 \neq y_2$'],'interpreter','latex')
axis image;colorbar; 



