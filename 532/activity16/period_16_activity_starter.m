%% Problem 1
clear
close all

%% Problem 1b

U = [1 0; 0 1; 0 0; 0 0];
S = [1 0; 0 0.5];
V = [1 0; 0 1];
X = U*S*V';

y = [1; 1/2; 1; 0];

w = V*inv(S)*U'*y;

disp('Optimum Weights')
disp(w)

c = y'*y -y'*X*w;


%% Find values for contour plot of surface

x1 = [-1:.01:3];  % range -1 to 3 
x2 = [-1:.01:3];

for i =1:length(x2)
    for j = 1:length(x1)
        t = [ x1(j);x2(i) ];
        z(i,j) = (t-w)'*X'*X*(t-w) +c;
    end
end

%% Plot values of coutour 
contour(x1,x2,z,20)
legend('Cost Function')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('1b')
axis square

%% Problem 1c

%% Problem 1d





%% Problem 2 

%% Gradient descent examples

%% First case (parts 2a - 2c)

U = [1 0; 0 1; 0 0; 0 0];
S = [1 0; 0 0.5];
V = 1/sqrt(2)*[1 1; 1 -1];
X = U*S*V';

y = [sqrt(2); 0; 1; 0];

tau = .5;

%% Find optimum w

w = V*inv(S)*U'*y;

disp('Optimum Weights')
disp(w)

c = y'*y -y'*X*w;

%% Find values for contour plot of surface

x1 = [-1:.01:3];  % range -1 to 3 
x2 = [-1:.01:3];

for i =1:length(x2)
    for j = 1:length(x1)
        t = [ x1(j);x2(i) ];
        z(i,j) = (t-w)'*X'*X*(t-w) +c;
    end
end

%% Find and display weights generated by gradient descent

w1 = [1.5;-.5];
it = 20;

W = graddescent(X,y,tau,w1,it);

figure
contour(x1,x2,z,20)
hold on
plot(w(1),w(2),'s', W(1,:),W(2,:),'o-','linewidth',2)
legend('Cost Function','Optimum Weights','Gradient Descent')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('\tau = 0.5')
axis square

%%  2b) 
w1 = [0;0];
it = 20;

W = graddescent(X,y,tau,w1,it);

figure
contour(x1,x2,z,20)
hold on
plot(w(1),w(2),'s', W(1,:),W(2,:),'o-','linewidth',2)
legend('Cost Function','Optimum Weights','Gradient Descent')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('\tau = 0.5')
axis square

%% 2b) continued
w1 = [0;2];
it = 20;

W = graddescent(X,y,tau,w1,it);

figure
contour(x1,x2,z,20)
hold on
plot(w(1),w(2),'s', W(1,:),W(2,:),'o-','linewidth',2)
legend('Cost Function','Optimum Weights','Gradient Descent')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('\tau = 0.5')
axis square

%% 2c)  

w1 = [1.5;-.5];
it = 20;
tau = 2.5;

W = graddescent(X,y,tau,w1,it);

figure
contour(x1,x2,z,20)
hold on
plot(w(1),w(2),'s', W(1,:),W(2,:),'o-','linewidth',2)
legend('Cost Function','Optimum Weights','Gradient Descent')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('\tau = 2.5')
axis square

%% 2d)
U = 0 %fill me in 
S = 0 % fill me in 
V = 0 %fill me in 
X = U*S*V';

y = 0 %fill me in 
tau = 0%fill me in 

%% Find optimum w

w = V*inv(S)*U'*y;

disp('Optimum Weights')
disp(w)

c = y'*y -y'*X*w;

%% Find values for contour plot of surface

x1 = [-1:.01:3];  % range -1 to 3 
x2 = [-1:.01:3];

for i =1:length(x2)
    for j = 1:length(x1)
        t = [ x1(j);x2(i) ];
        z(i,j) = (t-w)'*X'*X*(t-w) +c;
    end
end

%% Find and display weights generated by gradient descent

w1 = 0 %fill me in 
it = 0 %fill me in 

W = graddescent(X,y,tau,w1,it);

figure
contour(x1,x2,z,20)
hold on
plot(w(1),w(2),'s', W(1,:),W(2,:),'o-','linewidth',2)
legend('Cost Function','Optimum Weights','Gradient Descent')
ax = gca; % current axes
ax.FontSize = 14;
xlim([-1,3])
xlabel('w_1')
ylim([-1,3])
ylabel('w_2')
title('\tau = 0.5')
axis square


