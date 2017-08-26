function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% m is the number of training examples
X = [ones(m, 1), X];
hidden_layer = sigmoid(Theta1 * X');

hidden_layer = [ones(m, 1), hidden_layer'];
output_layer = sigmoid(Theta2 * hidden_layer');

% [Y,I] = max(X) returns the indices of the maximum values in vector I.
% maxPred and maxRowIndex dimension: 1 by number_of_training_examples
% maxPred is the max probability predicted by logistic regression
% maxRowIndex is the row index of max probability corresponding to the
% class label

[maxPred, maxRowIndex] = max(output_layer);

p = maxRowIndex';




% =========================================================================


end
