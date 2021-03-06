% Date:     19/2/2015
% Title:    Feed Forward One Layer Neural Network
% Author:   Daniel Dixey

% Developed: 
%   From reading the literature

% Purpose: 
%   Use previous matlab experiance
%   Understand the mechanics of the neural network mathematically

% Description:
%   Attempting to build a feed forward - Successful
%   A one layer Neural Network where nInput = nOutputs

% Clear any History and Workspace Objects
close all;
clear;
clc;

% Create all the necassary matrices for building a one layer feed forward
% neural network
[nInputs,nHiddenNeurons,nOutputs,hidWeights,outWeights, hActiv,...
            outActiv,inOutput,hidOutput,outOutput, outChange, hidChange] = ...
                                    one_layer_setup(2, 2, 1);

% Training the Neural Network:
epoch = 0;
for epochN = 1:100;
    % Define the learning Rate
    learn_rate = 0.1;
    
    % Loop through each example once
    inputs = ([[0,0];[0,1];[1,0];[1,1]]);
    
    % Define the targets
    targets = ([[0];[1];[1];[0]]);
    
    for sub = 1:size(inputs,1);
        % Get input
        input = inputs(sub,:);
        
        % Get Target Value
        target = targets(sub,:);
        
        % Feed forward through the network
        [inOutput, hActiv, hidOutput, outActiv, outOutput] = ...
            feed_forward(hidWeights,outWeights,hActiv,outActiv, ...
            inOutput,hidOutput,outOutput, input);
        
        % Adjust Weights in a Backward Direction
        [error, outChange, hidChange, hidWeights, outWeights] = ...
            changing_weights(target, learn_rate, outOutput, outChange, outActiv, hidChange, ...
            hActiv, outWeights, hidWeights, inOutput, hidOutput);
    end
    fprintf('Finish Epoch %f\n', epochN);
    fprintf('Current Error %f\n\n', error);
end
