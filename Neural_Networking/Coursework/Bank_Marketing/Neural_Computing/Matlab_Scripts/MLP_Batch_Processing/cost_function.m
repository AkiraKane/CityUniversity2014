function Cost = cost_function(Target, NN_Output)
    % Function for find the Error in the Neural Network
    
    % NN_Outpt is the Prediction values of the NN
    % Target is the desired output of the NN
    
    % Check dimension are correct
    if size(Target) ~= size(NN_Output)
        if size(Target,1) ~= size(NN_Output,1)
            error('Incorrect Number of Predictions')
        else
            error('Wrong Number of Prediction Classes')
        end
    end
    
    % Calculate the Cost Function Value - Mean Square Error
    % Cost = IMPLEMENT THE MSE CALCULATION
    
    % Calculate the Cost Function Value - CROSS ENTROPY FORMULA
    Cost = (-1.0/length(Target))*sum((Target.*log(NN_Output+realmin)) + ((1-Target).*log((1-NN_Output)+realmin)));