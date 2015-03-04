clc                                                         % Clear Command Window
clear all                                                   % Clear Workspace

Layer_1 = 100;                                               % Number of Neurons of the First Layer
Layer_2 = 100;                                                % Number of Neurons of the Second Layer
Learning_rate = 5;                                       % Learning rate of the Neural Net

Adjustment_1 = 10;
Adjustment_2 = 10;

Parameters = [Layer_1 Layer_2 Learning_rate Adjustment_1 Adjustment_2];               % Learning Rate of the Algorithm

% Results = fminsearch(@Cross_Validation, Parameters)         % Minimisation Function

Results = Cross_Validation( Parameters)         % Minimisation Function