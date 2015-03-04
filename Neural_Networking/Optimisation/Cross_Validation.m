function [Error] = Cross_Validation(configuration, X)
    Acc = 0;                                                % Avg Accuracy Reset
    MSE = 0;                                                % Avg MSE Reset
    k = 10;                                                 % Number of Cross Validation Folds to Use
    Indices = crossvalind('Kfold', length(X), 10);          % Get Indices for Cross Validation
    for i = 1:k;                                            % Loop through each fold
        fprintf('\n Cross Validation Fold %d...\n', i)      % Print Statement to Console
        Validation_Set = X(Indices == i, 1:size(X,2)-1);    % Create the Validation Data
        Validation_Set_Labels = X(Indices == i,size(X,2));  % Create the Validation Labels
        Training_Set = X(Indices ~=i, 1:size(X,2)-1);       % Create the Training Data
        Training_Set_Labels = X(Indices ~=i, size(X,2));    % Create the Training Labels
        configuration.sNum = size(Training_Set, 1);         % Sample Size
    
        [MSE_m, model, Acc_T] = train_nn(configuration,[],[], Training_Set, Training_Set_Labels, Validation_Set, Validation_Set_Labels);
        % Training the Model Function
    
        MSE =  MSE + MSE_m;                                 % Add new MSE on
        Acc = Acc + Acc_T;                                  % Add new Acc on
    end
    fprintf('\n\nAvg MSE for Each Fold %.8f and Avg Accuracy %.8f\n', MSE/k, Acc/k)
                                                            % Print an Overview of Results
    fprintf('Setup - Learning Rate = %.3f and Hidden Neurons = %d\n\n', Learning_rate, Neurons)
                                                            % Print an Overview of Results
                                                            
    Error = 1 - Acc;                                        % Get the Error

