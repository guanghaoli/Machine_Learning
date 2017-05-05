function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

a = X(y==1,:);
b = X(y==0,:);
scatter(a(:,1),a(:,2),40,'k+','LineWidth',2);
hold on;
scatter(b(:,1),b(:,2),40,'fill','yo','MarkerEdgeColor','k');
hold off;








% =========================================================================



hold off;

end
