function [Error] = Cross_Validation(Parameters)

    %configuration.hidNum = [floor(Parameters(1)/10), floor(Parameters(2)/10)];      % Number of Hidden Nodes by layer
    configuration.hidNum = [10, 10];      % Number of Hidden Nodes by layer
    configuration.activationFnc = {'tansig','tansig','logsig'}; % Activation Functions
    configuration.eNum   = 40;                                  % Max Number of Epochs to Train On
    configuration.bNum   = 1;                                   % Batch Size
    %configuration.params = [Parameters(3)/100 0.1 0.1 0.0001];      % Additional parameters
    configuration.params = [0.5 0.1 0.1 0.0001];      % Additional parameters

    file = 'bank-additional-full-encoded';             % Filename to Import
    X = csvread([file '.csv'],1,0);                             % Import the File into an Matrix

    %fprintf('\nNeurons on Layer 1: %d - Layer_2: %d and Learning Rate: %.9f\n', floor(Parameters(1)/10), floor(Parameters(2)/10), Parameters(3)/100)
                                                                % Display Tuning Results
    
    t = X(:, size(X,2));
    X = X(:, 1:size(X,2)-1);
                                                                
    % Implementing Smote (synthetic minority over sampling technique)
    fprintf('\nPre SMOTE Dataset\n')
    tabulate(t)
    
    folder = cd;

    % Adding necessary files to the java path
    javaaddpath([folder '/weka.jar']);
    javaaddpath([folder '/smote.jar']);
    
    smote = 1; % SMOTE oversampling

    if(smote),
        smoteStr = 'yes';
    else
        smoteStr = 'no';
    end;
    
    % Setting SMOTE up
    oldInputSize = size(X,1);
    if(smote),
        data = [X , t];
        
        data2arff(data, file);
        
        filter = javaObjectEDT('weka.filters.supervised.instance.SMOTE');
        
        fileReader = javaObjectEDT('java.io.FileReader',[file '.arff']);
        buffReader = javaObjectEDT('java.io.BufferedReader', fileReader);
        arffReader = javaObjectEDT('weka.core.converters.ArffLoader$ArffReader', buffReader);
        
        instances = arffReader.getData();
        instances.setClassIndex(size(data, 2)-1);
        
        paramFilter = java.lang.String('-C 0 -K 5 -P 100.0 -S 123');
        paramFilter = paramFilter.split(' ');
        
        filter.setOptions(paramFilter);
        filter.setInputFormat(instances);
        result = filter.useFilter(instances, filter);
        
        filteredData = zeros(result.numAttributes(),result.numInstances());
        for i = 1:result.numAttributes(),
            filteredData(i,:) = result.attributeToDoubleArray(i-1);
        end;
        
        
        t = (filteredData(size(filteredData, 1),:)*2)-1;
        fprintf('\nPost SMOTE Result\n')
        tabulate(t)
        X = filteredData;
        X(size(filteredData, 1),:) = [];
        
%         if(M == N),
%             N = size(X,2);
%             M = N;
%         else
%             N = size(X,2);
%         end;
    end;
    
    X = [X', t'];
                                                                
    Acc = 0;                                                    % Avg Accuracy Reset
    MSE = 0;                                                    % Avg MSE Reset
    k = 2;                                                     % Number of Cross Validation Folds to Use
    Indices = crossvalind('Kfold', length(X), k);              % Get Indices for Cross Validation
    for i = 1:k;                                                % Loop through each fold
        fprintf('\n Cross Validation Fold %d...\n', i)          % Print Statement to Console
        Validation_Set = X(Indices == i, 1:size(X,2)-1);        % Create the Validation Data
        Validation_Set_Labels = X(Indices == i,size(X,2));      % Create the Validation Labels
        Training_Set = X(Indices ~=i, 1:size(X,2)-1);           % Create the Training Data
        Training_Set_Labels = X(Indices ~=i, size(X,2));        % Create the Training Labels
        configuration.sNum = size(Training_Set, 1);             % Sample Size
        
        fprintf('\nDisplay and Overview of the Training Data Label\n\n')
        tabulate(Training_Set_Labels)
        
        [MSE_m, model, Acc_T] = train_nn(configuration,[],[], Training_Set, Training_Set_Labels, Validation_Set, Validation_Set_Labels);
        % Training the Model Function
    
        MSE =  MSE + MSE_m;                                     % Add new MSE on
        Acc = Acc + Acc_T;                                      % Add new Acc on
    end
    fprintf('\n\nAvg MSE for Each Fold %.8f and Avg Accuracy %.8f\n', MSE/k, Acc/k)
                                                                % Print an Overview of Results
                                                           
    Error = 1 - Acc/k;                                          % Get the Error

