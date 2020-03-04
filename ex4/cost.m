function [ J ] = cost( nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda )
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

mapy = zeros(length(y), num_labels);
for i = 1:length(y)
    mapy(i, y(i)) = 1;
end
a1 = [ones(size(X, 1),1) X]';
z2 = Theta1 * a1;
a2 = [ones(1, size(z2,2)); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
n = size(a3, 1);

J = mean(sum( - mapy' .* log(a3) + (mapy' -1) .* log(1 - a3)));
J = J + lambda * 0.5 / size(X, 1) * (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)));

end

