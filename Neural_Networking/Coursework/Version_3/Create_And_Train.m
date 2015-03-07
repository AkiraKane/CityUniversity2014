function Result_Line = Create_And_Train(X, X_target, Y, Y_target)
   
    % Get characteristics
    N_inputs = size(X, 2);
    N_outputs = size(X_target, 2);
    
    % Define the Number of Hidden Neurons
    N_Layers_N_Neurons = [2]; % Can be a row Vector
    
    % Create the Neural Network
    net = create_neural_network(N_inputs, N_outputs, N_Layers_N_Neurons);
    
    % Show Config
    show_setup(net)
    
    % Neural Network Parameters
    Reg     = 1e-5; % Regularisation Value
    epochs  = 30000;  % Number of Epochs to Train Network with
    LR      = 0.5;  % Learning Rate value
    Mo      = 0.1;  % Momentum
    % Define Stopping Criteria.....
        
    % Train Neural Network
    net = train_nn(net, X, Y, Reg, epochs, LR, Mo, X_target, Y_target);
    
    % Final Evaluation
    [net, Y_Prediction_t] = forward_propagate(X, net);
    [net, Y_Prediction_v] = forward_propagate(Y, net);
    
    % Calculate confusion matrix - Training 
    [c,cm,ind,per] = confusion(X_target',round(Y_Prediction_t)');
    
    % Calculating Results - Training
    Accuracy    = (cm(1,1) + cm(2,2)) / sum(sum(cm));
    Sensitivity = cm(1,1) / (cm(1,1) + cm(1,2));
    Specificity = cm(2,2) / (cm(2,2) + cm(2,1));
    Precision   = cm(1,1) / (cm(1,1) + cm(2,1));
    
    % Training Statistics
    Training = [Accuracy, Sensitivity, Specificity, Precision];
    
    % Calculate confusion matrix - Validation 
    [c,cm,ind,per] = confusion(X_target',round(Y_Prediction_v)');
    
    % Calculating Results - Training
    Accuracy    = (cm(1,1) + cm(2,2)) / sum(sum(cm));
    Sensitivity = cm(1,1) / (cm(1,1) + cm(1,2));
    Specificity = cm(2,2) / (cm(2,2) + cm(2,1));
    Precision   = cm(1,1) / (cm(1,1) + cm(2,1));
    
    % Validation Statistics
    Validation = [Accuracy, Sensitivity, Specificity, Precision];
    
    % Plot confusion matrix
    %plotconfusion(X_target',round(Y_Prediction_t)')
    
    % Get Results
    Result_Line = [LR, epochs, Reg, Mo, Training, Validation];