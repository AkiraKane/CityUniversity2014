function [net, Prediction] = forward_propagate(Data, net)
    % Forward Propagation
    
    % Data  - Input Data, can be either Training or Test
    % net   - The Neural Network Model
    
    % Number of Layers excluding the Input layer
    T = length(net.Weights);
    
    % Storing the Activations
    net.Input = Data;
    for i = 1:T+1
        net.Activation{i} = zeros(1,1);
    end
    
    
    % Forward Propagation
    for i=1:T
        
        % Check if there is sufficent Data
        if size(net.Activation{i}, 2) == 1
            net.Activation{i} = reshape(net.Activation{i+1},[1,size(net.Activation{i+1},1)]);
        end
        
        if i == 1
            % Add bias values onto the 'Inputs'
            net.Activation{i} = [ones(size(net.Input,1),1) net.Input];
        else
             % Add Bias values on the Activation
            net.Activation{i} = [ones(size(net.Activation{1}, 1),1) net.Activation{i}];
        end
       
        % Multiply the Weight by Inputs
        W_by_Inputs = net.Activation{i}*net.Weights{i}';
        % Add Results by Activation Function to Next value
        net.Activation{i+1} = arrayfun(@activation, W_by_Inputs);
    end
    
    % Provide Network Output
    Prediction = net.Activation{T+1};
    
    
    
    