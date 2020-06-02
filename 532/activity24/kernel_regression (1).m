%% Kernel Curve Fitting

close all
clear

sigma = 0.04; % Gaussian kernel "width"
lam = 0.01; % Ridge regression parameter
n = 50; % number of data points
rng('default'); % restart random number seed for identical data each run

%% Generate example data

x = rand(n,1);

d = .4*sin(1.5*pi*x) + x.^2 + .04*randn(n,1);

%% Display example kernels

xtest = 0:.01:1; % uniformly sample interval 0 to 1

for i=1:length(xtest)
    for j=1:20:length(xtest) % find kernels for every 20th center
        Kdisplay(i,j) = exp(-(xtest(i)-xtest(j))^2/(2*sigma^2));
    end
end

figure
plot(xtest,Kdisplay,'linewidth',2)
xlabel('x')
ylabel('Kernel value')
title(['Example Kernels for \sigma = ',num2str(sigma,2)])
ax = gca;
ax.FontSize = 14;
        
%% Kernel fitting

distsq=zeros(n,n);
for i =1:n
    for j=1:n
        distsq(i,j) = (x(i)-x(j))^2;
    end
end

K = exp(-distsq/(2*sigma^2));


% Find alpha

alpha = inv(K+lam*eye(size(K)))*d;

%% Predict Curve

distsqtst = zeros(length(xtest),n);
for i = 1:length(xtest)
    for j = 1:n
        distsqtst(i,j) = (xtest(i)-x(j))^2;
    end
end

dtest = exp(-distsqtst/(2*sigma^2))*alpha;

dtrue = .4*sin(1.5*pi*xtest) + xtest.^2; % noise free

figure
plot(x,d,'x',xtest,dtest,'r',xtest,dtrue,'g','linewidth',2)
legend('Measured data','Kernel fit','True noise free')
title(['Data and Kernel Fit, \lambda = ',num2str(lam,2), ', \sigma = ',num2str(sigma)])
xlabel('x')
ylabel('d')
ax = gca;
ax.FontSize = 14;

