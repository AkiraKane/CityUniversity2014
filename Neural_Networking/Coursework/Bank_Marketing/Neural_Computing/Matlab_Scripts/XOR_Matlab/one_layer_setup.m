function [nInputs,nHiddenNeurons,nOutputs,hidWeights,outWeights, hActiv,...
            outActiv,inOutput,hidOutput,outOutput, outChange, hidChange] = ...
                                one_layer_setup(In, Hidden, Out)
    % Number of Neurons in each Layer
    nInputs = In;
    nHiddenNeurons = Hidden;
    nOutputs = Out;

    % Initialize Weights with random values
    hidWeights = rand(nHiddenNeurons, nInputs + 1);
    outWeights = rand(nOutputs, nHiddenNeurons + 1);

    % Create Empty vectors for the Activation Calculations
    hActiv = zeros(nHiddenNeurons, 1);
    outActiv = zeros(nOutputs ,1);

    % Outputs of the Neurons
    inOutput = zeros(nInputs + 1, 1);
    hidOutput = zeros(nHiddenNeurons + 1, 1);
    outOutput = zeros(nOutputs);
    
    % Change in Weights for the Hidden and Output Layers
    outChange = zeros(nHiddenNeurons);
    hidChange = zeros(nOutputs);
end