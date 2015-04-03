function Result_Line = Create_And_Train(X, X_target, Y, Y_target, epochs, LR, Mo, Reg, Hidden_Neurons, Show_Config)
   
    % Get characteristics
    N_inputs = size(X, 2);
    N_outputs = size(X_target, 2);
    
    % Define the Number of Hidden Neurons
    N_Layers_N_Neurons = [Hidden_Neurons]; % Can be a row Vector
    
    % Create the Neural Network
    net = create_neural_network(N_inputs, N_outputs, N_Layers_N_Neurons);
    
    % Show Config
    if Show_Config == 1;
        show_setup(net)
    end
    
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
    Cost = cost_function(X_target, Y_Prediction_t);
    
    % Training Statistics
    Training = [Accuracy, Sensitivity, Specificity, Precision Cost];
    
    % Calculate confusion matrix - Validation 
    [c,cm,ind,per] = confusion(Y_target',round(Y_Prediction_v)');
     
    % Calculating Results - Validation
    Accuracy    = (cm(1,1) + cm(2,2)) / sum(sum(cm));
    Sensitivity = cm(1,1) / (cm(1,1) + cm(1,2));
    Specificity = cm(2,2) / (cm(2,2) + cm(2,1));
    Precision   = cm(1,1) / (cm(1,1) + cm(2,1));
    Cost = cost_function(Y_target, Y_Prediction_v);
    
    % Validation Statistics
    Validation = [Accuracy, Sensitivity, Specificity, Precision Cost];
    
    % Plot confusion matrix - ADD FUNCTION TO SAVE TO PNG/ JPEG file
    %plotconfusion(X_target',round(Y_Prediction_t)')
    
    % Get Results
    Result_Line = [Training, Validation];