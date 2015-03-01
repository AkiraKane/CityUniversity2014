clc                                                         % Clear Command Window
clear all                                                   % Clear Workspace


Combins = allcomb([5:5:50],[5:5:95]);                      % Using the allcomb function (LICENSE IN FOLDER)
                                                            % Changing Learning Rate and number of Epochs
                                                       
file = 'bank-additional-full-encoded';                      % Filename to Import
X = csvread([file '.csv'],1,0);                             % Import the File into an Matrix

store_Results = NaN(size(Combins,1),5);                     % Store Results
    
for row = 1:size(Combins,1);                                % Iterate through combinations
    Learning_rate = Combins(row,2)/100;                     % Learning Rate from Combinations
    Neurons = Combins(row,1);                               % Number of Neurons at Each Layer
    
                                                            % Configurable Parameters
    configuration.hidNum = [Neurons, Neurons, Neurons];     % Number of Hidden Nodes by layer
    configuration.activationFnc = {'tansig','tansig','tansig','logsig'}; % Activation Functions
    configuration.eNum   = 10;                             % Max Number of Epochs to Train On
    configuration.bNum   = 1;                               % Batch Size
    configuration.params = [Learning_rate 0.1 0.1 0.0001];  % Additional parameters
    
    
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
    
    store_Results(row, :) = [row, MSE/k, Acc/k, Neurons, Learning_rate];
                                                              % Store results in Matrix
end

minMSE = min(store_Results(:,2));
maxAcc = max(store_Results(:,3));

plot3(store_Results(:,4) , store_Results(:,5), store_Results(:,2))

