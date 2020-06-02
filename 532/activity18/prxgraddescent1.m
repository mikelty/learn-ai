function [W,Z] = prxgraddescent1(X,y,tau,lam,w1,it)

% compute it iterations of L1 proximal gradient descent starting at w1
%
%
% step size tau

W(:,1) = w1;

for k = 1:it
    Z(:,k+1) = W(:,k) - tau*X'*(X*W(:,k) - y);
    W(:,k+1) = wthresh(Z(:,k+1),'s',lam*tau/2);
end

