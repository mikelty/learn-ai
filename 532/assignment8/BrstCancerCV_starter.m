%% Breast Cancer LASSO Exploration
%% Prepare workspace

close all
clear

load BreastCancer

%%  10-fold CV 

% each row of setindices denotes the starting an ending index for one
% partition of the data: 5 sets of 30 samples and 5 sets of 29 samples
setindices = [1,30;31,60;61,90;91,120;121,150;151,179;180,208;209,237;238,266;267,295];

% each row of holdoutindices denotes the partitions that are held out from
% the training set
holdoutindices = [1,2;2,3;3,4;4,5;5,6;7,8;9,10;10,1];

cases = size(holdoutindices,1);

% be sure to initiate the quantities you want to measure before looping
% through the various training, validation, and test partitions
%
% 
%

% Loop over various cases
for j = 1:cases
    % row indices of first validation set
    v1_ind = setindices(holdoutindices(j,1),1):setindices(holdoutindices(j,1),2);
    
    % row indices of second validation set
    v2_ind = setindices(holdoutindices(j,2),1):setindices(holdoutindices(j,2),2);
    
    % row indices of training set
    trn_ind = setdiff(1:295,[v1_ind, v2_ind]);
    
    % define matrix of features and labels corresponding to first
    % validation set
    Av1 = X(v1_ind,:);
    bv1 = y(v1_ind);
    
    % define matrix of features and labels corresponding to second
    % validation set
    Av2 = X(v2_ind,:);
    bv2 = y(v2_ind);
    
    % define matrix of features and labels corresponding to the 
    % training set
    At = X(trn_ind,:);
    bt = y(trn_ind);

% Use training data to learn classifier
%   W = ista_solve_hot(At,bt,lam_vals);
%
% Find best lambda value using first validation set, then evaluate
% performance on second validation set, and accumulate performance metrics
% over all cases partitions
    
end




    
    
    
    
