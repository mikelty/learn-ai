function W = graddescent(X,y,tau,w1,it)

% compute 10 iterations of gradient descent starting at w1
%
% w_{k+1}= w_k - tau*X'*(X*w_k - y)
%
% step size tau

W(:,1) = w1;

for k = 1:it
    W(:,k+1) = W(:,k) - tau*X'*(X*W(:,k) - y);
end

