function [W,Z] = prxgraddescent2(X,y,tau,lam,w1,it)

% compute it iterations of L2 proximal gradient descent starting at w1
%
% w_{k+1}= (w_k - tau*X'*(X*w_k - y)/(1+lam*tau)
%
% step size tau

W(:,1) = w1;

for k = 1:it
    Z(:,k+1) = W(:,k) - tau*X'*(X*W(:,k) - y);
    W(:,k+1) = Z(:,k+1)/(1+lam*tau);
end

