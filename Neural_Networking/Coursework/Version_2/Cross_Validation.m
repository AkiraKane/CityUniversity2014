function [Error] = Cross_Validation(Parameters)

    configuration.hidNum = [floor(Parameters(1)/10 + (1/100)*Parameters(4)), floor(Parameters(2)/10  + (1/100)*Parameters(5))];      % Number of Hidden Nodes by layer
    configuration.activationFnc = {'tansig','tansig','logsig'}; % Activation Functions
    configuration.eNum   = 40;                                  % Max Number of Epochs to Train On
    configuration.bNum   = 1;                                   % Batch Size
    configuration.params = [Parameters(3)/100 0.1 0.1 0.0001];      % Additional parameters

    file = 'bank-additional-full-binary-transform';             % Filename to Import
    X = csvread([file '.csv'],1,0);                             % Import the File into an Matrix

    fprintf('\nNeurons on Layer 1: %d - Layer_2: %d and Learning Rate: %.9f Adjustment Factor 1: %.4f 2: %.4f \n', floor(Parameters(1)/10), floor(Parameters(2)/10), Parameters(3)/100, Parameters(4), Parameters(5))
                                                                % Display Tuning Results
    
    Acc = 0;                                                    % Avg Accuracy Reset
    MSE = 0;                                                    % Avg MSE Reset
    k = 1;                                                     % Number of Cross Validation Folds to Use
    Indices = crossvalind('Kfold', length(X), 10);              % Get Indices for Cross Validation
    for i = 1:k;                                                % Loop through each fold
        fprintf('\n Cross Validation Fold %d...\n', i)          % Print Statement to Console
        Validation_Set = X(Indices == i, 1:size(X,2)-1);        % Create the Validation Data
        Validation_Set_Labels = X(Indices == i,size(X,2));      % Create the Validation Labels
        Training_Set = X(Indices ~=i, 1:size(X,2)-1);           % Create the Training Data
        Training_Set_Labels = X(Indices ~=i, size(X,2));        % Create the Training Labels
        configuration.sNum = size(Training_Set, 1);             % Sample Size
        
        fprintf('\nDisplay and Overview of the Training Data Label\n\n')
        tabulate(Training_Set_Labels)
        
        [MSE_m, model, Acc_T] = train_nn(configuration,[],[], Training_Set, Training_Set_Labels, Validation_Set, Validation_Set_Labels);
        % Training the Model Function
    
        MSE =  MSE + MSE_m;                                     % Add new MSE on
        Acc = Acc + Acc_T;                                      % Add new Acc on
    end
    fprintf('\n\nAvg MSE for Each Fold %.8f and Avg Accuracy %.8f\n', MSE/k, Acc/k)
                                                                % Print an Overview of Results
                                                           
    Error = 1 - Acc/k;                                          % Get the Error

