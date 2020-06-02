%% Neural Net Example
%% Set up workspace
close all
clear

p = 2;  % features
n = 1e4;  % examples

%% Generate training data

X = rand(n,p)-.5;  % n examples of p features
Y1 = sum(X.^2,2)>.1;  % labels for first class problem
Y2 = 5*X(:,1).^3 > X(:,2);  % labels for 2nd class problem
Y = [Y1 Y2];

% display features and labels
figure(1);clf;

% First classification problem
subplot(121);
scatter(X(:,1),X(:,2),20,Y1,'filled');
title('training data, label 1'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)

% Second classification problem
subplot(122);
scatter(X(:,1),X(:,2),20,Y2,'filled');
title('training data, label 2'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)


%% Train neural network

Xb = [ones(n,1) X]; % append constant term as first column

q = size(Y,2); % number of classification problems (output nodes)
M = 4; % number of hidden nodes

% Initial guesses on weights
V = randn(M+1,q); % weights from hidden layer to output
W = randn(p+1,M); % weights from input to hidden layer

alpha = .1;  % step size

L = 10;
for epoch = 1:L % make L passes through all training samples
    
   ind = randperm(n); % random permutation of integers 1:n (SGD)
   
   for i = ind % loop over all training samples in order given in ind
       
      % forward propagation
      H = logsig([1 Xb(i,:)*W]); % 1 x M+1 output of hidden layer
      Yhat = logsig(H*V); % 1 x q output nodes
      
      % back propagation
      delta = (Yhat-Y(i,:)).*Yhat.*(1-Yhat); % 1 x q
      Vnew = V-alpha*H'*delta; 
      
      gamma = (delta*V(2:end,:)').*H(2:end).*(1-H(2:end)); % 1 x M
      Wnew = W-alpha*Xb(i,:)'*gamma;
      
      V = Vnew;
      W = Wnew;
      
   end
   
   epoch
end
%% Final predicted labels

H = logsig([ones(n,1) Xb*W]); % n x M+1 output of hidden layer
Yhat = logsig( H*V); % n x q outputlayer

% Display results
figure(2);clf;

% First classification problem (first output node)
subplot(121);scatter(X(:,1),X(:,2),20,Yhat(:,1),'filled');
title('learned labels, label 1'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)

% Second classification problem (second output node)
subplot(122);scatter(X(:,1),X(:,2),20,Yhat(:,2),'filled');
title('learned labels, label 2'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)

% Display thresholded learned labels
figure(3);clf;

% First classification problem (first output node)
subplot(121);scatter(X(:,1),X(:,2),20,1*(Yhat(:,1)>.5),'filled');
title('thresh labels, label 1'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)

% Second classification problem (second output node)
subplot(122);scatter(X(:,1),X(:,2),20,1*(Yhat(:,2)>.5),'filled');
title('thresh labels, label 2'); 
axis image;colorbar;colormap jet; set(gca,'fontsize',18)

% Plot misclassified test data

figure(4);clf;

err = abs((Yhat(:,1)>.5)-Y(:,1));
subplot(121); scatter(X(:,1),X(:,2),20,err,'filled');colormap cool
set(gca,'fontsize',18)
title([num2str(sum(err)),' Errors $\hat{y}_1 \neq y_1$'],'interpreter','latex')
axis image;colorbar; 

err = abs((Yhat(:,2)>.5)-Y(:,2));
subplot(122); scatter(X(:,1),X(:,2),20,err,'filled');colormap cool
set(gca,'fontsize',18)
title([num2str(sum(err)),' Errors $\hat{y}_2 \neq y_2$'],'interpreter','latex')
axis image;colorbar; 

