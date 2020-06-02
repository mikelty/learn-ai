%% Activity 4.3 PageRank
%% Prepare workspace

close all
clear

%% Circle topology

% Define unweighted adjacency matrix.  Note semicolons denote the end of a
% row while ... tells MATLAB that the instruction is continued on the next
% line


Atilde = [ ; ; ; ...
    ; ; ;...
    ; ]

% Weighted adjacency - normalize by the sum of all elements in a column
for i = 1:8
    A(:,i) = Atilde(:,i)/sum(Atilde(:,i));
end

A

%% Power method

b0 = ones(8,1)/8;

b1 = ???

% 1000 iterations

b = b0;
for k = 1:1000
    b = ???;
end

b

%% Hub topology

% Define unweighted adjacency matrix

Atildehub = [ ; ; ; ...
    ; ; ;...
    ; ; ]

% Weighted adjacency
for i = 1:9
    Ahub(:,i) = Atildehub(:,i)/sum(Atildehub(:,i));
end

Ahub

%% Power method

b0 = ones(9,1)/9;

bhub1 = ??

% 100 iterations

bhub = b0;
for k = 1:1000
    bhub = ??;
end

bhub