%% Gradient descent examples
%% prepare workspace

close all
clear

%% Least Squares Problem


X = [2 1];

y = [4];

%% Find values for contour plot of surface

x1 = [-1:.01:3];  % range -1 to 3 
x2 = [-1:.01:3];

for i =1:length(x2)
    for j = 1:length(x1)
        t = [ x1(j);x2(i) ];
        z(i,j) = (X*t-4)^2;
    end
end

%% Find and display weights generated by gradient descent

winit = [0;0];
lam = 4;

it = 10;
tau = 0.25;

[W,Z] = prxgraddescent1(X,y,tau,lam,winit,it);

% Concatenate gradient and regularization steps to display trajectory
for i = 1:it
    G(:,2*(i-1)+1:2*i)= [W(:,i),Z(:,i+1)];
end
G(:,2*i+1) = W(:,it+1);

figure
[C,h] = contour(x1,x2,z,20);
% clabel(C,h)
hold on
plot( Z(1,2:it+1),Z(2,2:it+1),'bx', W(1,:),W(2,:),'ro',G(1,:),G(2,:),'-c','linewidth',2)
legend('Cost Function','Gradient Descent Step', 'Regularization Step')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title(['\tau = ',num2str(tau), ', \lambda = ',num2str(lam)])
axis('square')

%% Second Case

