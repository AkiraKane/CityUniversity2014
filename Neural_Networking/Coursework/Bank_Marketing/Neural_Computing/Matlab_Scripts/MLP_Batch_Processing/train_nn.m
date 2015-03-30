function [net] = train_nn(net, X, Y, Reg, epochs, LR, Mo, X_target, Y_target)
    % Inputs into the Network
    
    % net   - The initial network that has been set up
    % X     - The Training Data
    % Y     - Validation Data
    % Reg   - Regularization Term - Defaulted to 1x10^-5
    % epoch - Number of times the Net is trained with - Defaulted to 100
    % LR    - Learning Rate of the Network
    % Mo    - Momentum of the Network - Assistance for finding the Minima
    
    % fprintf('\nTraining Neural Neural Network\n\n') % Uncomment to Inspect
    
    % Delta Weights - Create Weight Changing Stroage
    for i = 1:length(net.Weights)       
        net.D_Weight{i} = rand(size(net.Weights{i},1), size(net.Weights{i}, 2));
    end
    
    % Create Cost Function Storage
    net.Cost_Function_Training   = zeros(epochs, 1);
    net.Cost_Function_Validation = zeros(epochs, 1);
    
    % Forward Propagate the Network
    [net, Y_Prediction_t]   = forward_propagate(X, net); % Forward Propagate the Training Dataset
    [net, Y_Prediction_v]   = forward_propagate(Y, net); % Forward Propagate the Test Dataset
    [net, Y_Prediction_t]   = forward_propagate(X, net); % Forward Propagate the Training Dataset
    
    % Calculate Cost Functions
    net.Cost_Function_Training(1,1)     = cost_function(X_target, Y_Prediction_t);
    net.Cost_Function_Validation(1,1)   = cost_function(Y_target, Y_Prediction_v);
    net.Cost_Function_Training(1,1)     = cost_function(X_target, Y_Prediction_t);
    
    % Proceed with Training - Starting with Backpropagation
    for e=2:epochs
        %fprintf('Epoch: %i\n', e)
        % Backpropage the Error through the Network
        net = back_propagation(net, X_target, Reg, Y_Prediction_t);

        % Calculate Delta Weights and Include Momentum
        for l = 1:length(net.Gradient)
           net.D_Weight{l}  = (LR * net.Gradient{l}) + (net.D_Weight{l} * Mo);
           net.Weights{l}   = net.Weights{l} - net.D_Weight{l};
        end
        
        % Update Units
        [net, Y_Prediction_t]   = forward_propagate(X, net);
        
        % Calculate confusion matrix - Validation 
        [c,cm,ind,per] = confusion(Y_target',round(Y_Prediction_v)');
     
        % Calculating Results - Validation
        cm
        
        % Record Cost Function Value
        net.Cost_Function_Training(e,1) = cost_function(X_target, Y_Prediction_t);   
                
    end
    
    %fprintf('Training Complete...\n')