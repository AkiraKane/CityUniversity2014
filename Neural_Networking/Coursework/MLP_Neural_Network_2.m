clc                                                         % Clear Command Window
clear all                                                   % Clear Workspace

Layer_1 = 40;                                               % Number of Neurons of the First Layer
Layer_2 = 10;                                                % Number of Neurons of the Second Layer
Learning_rate = 10;                                       % Learning rate of the Neural Net

Parameters = [Layer_1 Layer_2 Learning_rate];               % Learning Rate of the Algorithm

Results = fminsearch(@Cross_Validation, Parameters)         % Minimisation Function