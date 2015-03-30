function [net] = create_neural_network(N_inputs, N_outputs, N_Layers_N_Neurons)
    % Creating the Neural Network
    % N_inputs = Number of Input Dimensions
    % N_Outputs = Number of Dimensions on the Output Later
    % N_Layers_N_Neurons = A vector of the Number of Neurons at each layer
    
    % Create a default of the Number of hidden Neurons to initiate with
    if isempty(N_Layers_N_Neurons)
        N_Layers_N_Neurons = [2];
    end
    
    % Create Net Shell
    net = [];
       
    % Create a Matrix of the Setup
    Combined = [[N_inputs], [N_Layers_N_Neurons], [N_outputs]];
    
    % Initate Neurons with a Weight in Range [a,b]
    a = 20; b = -20;
    
    % Iterate through and create Weight Matrix(s)
    for i = 1:length(Combined)
        % Ignore the Input Now
        if i > 1
            net.Weights{i-1} = a + (b-a).*rand(Combined(:,i),Combined(:,i-1)+1);
        end
    end