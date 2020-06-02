clear
close all

edges = csvread('wisconsin_edges.csv');
node_count = max(edges(:))+1;

A = zeros(node_count,node_count);
[m,n] = size(edges);
for i=1:m
  from_node = edges(i,1)
  to_node = edges(i,2)
  A(to_node+1,from_node+1)=1;
end

% Hint: use 
% eigs(A,k)
% where k=1 to get the first eigenvector, instead of 
% eig(A)
% as computation of all eigenvectors will take ~5 minutes