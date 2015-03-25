function [error, outChange, hidChange, hidWeights, outWeights] = ...
    changing_weights(target, learn_rate, outOutput, outChange, outActiv, hidChange, ...
                        hActiv, outWeights, hidWeights, inOutput, hidOutput)
    % Calculate Output Layer Error
    error = outOutput - target;
    
    % Calculate the deltas of the Output Neurons (Working backwards)
    outChange = (1 - tanh(outActiv)) * tanh(outActiv) * error;
    
    % Calculate the deltas of the Hidden Neurons (Working backwards)
    hidChange = times((1-tanh(hActiv)), times(tanh(hActiv), (outWeights(:,1:(size(outWeights,2)-1))') * outChange));
    
    % Apply Change to Weights
    hidWeights = hidWeights - learn_rate * hidChange*inOutput';
    outWeights = outWeights - learn_rate * (outChange*hidOutput');
    
end