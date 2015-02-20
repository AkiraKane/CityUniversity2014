function [inOutput, hActiv, hidOutput, outActiv, outOutput] = ...
    feed_forward(hidWeights,outWeights,hActiv,outActiv, ...
    inOutput,hidOutput,outOutput, input)
    % Set the Input as the Outputs of the First layer
    inOutput(1:size(inOutput,1)-1,:) = input;
    inOutput(size(inOutput,1),:) = 1; % Add bias equal to One
    
    % Calculate the Output of the Activation Function at the Hidden Layer
    hActiv = hidWeights*inOutput;
    hidOutput = tanh(hActiv);
    hidOutput(size(hidOutput, 1)+1, :) = 1; % Add bias equal to One
    
    % Calculations at the Output Layer
    outActiv = outWeights*hidOutput;
    outOutput = tanh(outActiv);
   
end