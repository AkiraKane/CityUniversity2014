function net = back_propagation(net, X_target, Reg, Y_Prediction_t)
    % Function for Calculation of Backpropgation
   
    % Get Indices for use later
    T = length(net.Weights);
    X_Prediction_t = net.Activation{T+1};
    
    % Create a Temporary Array to Store values
    Temp = [];
    for i=1:T
        Temp.L_Delta{i} = ones(1,1);
    end
    
    delta = X_Prediction_t - X_target;
    % Reshape Dimensions if = 1
    if ndims(delta) == 1
        delta = reshape(delta, [1, size(delta, 1)]);
    end
    
    Temp.L_Delta{1} = delta;
    
    k = 2;
    for j=T:-1:1+1
        % Add Data to Temp Object
        Temp.L_Delta{k} = (Temp.L_Delta{k-1}*net.Weights{j}(:,2:end)).*(net.Activation{j}(:,2:end).*(1 - net.Activation{j}(:,2:end)));
        k = k + 1;
    end
    
    % Reverse Order for Calculation Simplicity
    Temp.L_Delta = fliplr(Temp.L_Delta);
    
    % Calculate Gradients
    for l=1:T
        Temp.Gradient{l} = Temp.L_Delta{l}'*net.Activation{l};
    end
    
    % Regularisation
    N_Obs = size(X_target ,1);
    for i = 1:length(net.Weights)
        Temp.regGradient{i} = [zeros(size(net.Weights{i},1),1) net.Weights{i}(:,2:end)];
        net.Gradient{i} = ((Temp.Gradient{i})*(1/N_Obs)) + (Reg * Temp.regGradient{i});
    end    
    
    
    
    
    
    
    
    