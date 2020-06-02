%% Low Rank Bucky
%% Prepare workspace

close all
clear
A = csvread('bucky.csv');
maxdim = min(size(A));

%% Display Bucky

figure(1); 
imagesc(A)
colormap gray; 
axis image; 
axis off

%% Bucky's Singular Values

% COMPLETE AND UNCOMMENT LINE 21!

%[u,s,v] = ... ;  % take svd of image

figure(2);
plot(log10(diag(s)),'linewidth',2);  % plot log base 10 of singular vals
xlabel('index'); ylabel('log of sing vals');
title('Bucky''s singular values')
ax=gca;
ax.FontSize = 14;

%% Low-Rank Approximation

r_vals = [10 20 50 100];  % values of r (rank) to consider

for i = 1:4
    r = r_vals(i); % pick one rank at a time
    
    % COMPLETE AND UNCOMMENT LINES 39 AND 34
    
   % Ar = ... ; % create low-rank approximation
   % Er = ... ;  % Approximation error matrix
    
    
    figure(i+2)
    imagesc(Ar,[0 1])
    colormap gray; axis image; axis off
    title(['Rank r = ' num2str(r)])
    
    err(i) = norm(Er,'fro'); % Frobenius norm of error
    
end

figure(7)
stem(r_vals, (err/norm(A,'fro')).^2,'linewidth',2) % normalized squared Fro norm
ax=gca;
ax.FontSize = 14;
xlabel('Rank')
ylabel('Fractional Squared Error')
title('Low-Rank Bucky Approximation Error')

%% Bias-Variance Tradeoff

singvals = diag(s);

for r = 1:maxdim;
    bias2(r) = singvals(r:end)'*singvals(r:end);
end

sigma2 = 10;
var = sigma2*(1:maxdim);

figure(8)
plot(1:maxdim,log10(bias2),'b',1:maxdim,log10(var),'g',1:maxdim,log10(bias2+var),'r','linewidth',2)
ax=gca;
ax.FontSize = 14;
xlabel('Rank of Approximation')
ylabel('log_{10} Squared Errors')
legend('Bias Squared','Variance','Bias Squared plus Variance')


    