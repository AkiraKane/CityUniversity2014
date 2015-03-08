function y = derivative(x)
    % Derivative of the Sigmoid Function
    y = sigmoid(x) * (1 - sigmoid(x));