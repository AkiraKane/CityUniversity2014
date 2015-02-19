%% IMPORTING DATA

close all; clear all; clc;

[folder, name, ext] = fileparts(which('nnTrain'));
cd(folder);

% Adding necessary files to the java path
javaaddpath([folder '/weka.jar']);
javaaddpath([folder '/smote.jar']);

% Preparing the data
file = 'promoter';
X = csvread([file '.csv'], 0, 0);

t = X(:,size(X,2));
X(:,size(X,2)) = [];

Type = 'scaling';
%This function normalizes each column of an array  using the standard score or feature scaling.
X = StatisticalNormaliz(X,Type);
X = (X * 2) - 1; 
X = X';

E = 1;
originalInputSize = size(X,2);
classes = [1 -1];

% Get Generator
st = rand('state');

%% INPUT VARIABLES                                                                                                                                                     
% Network Init
K = 2; % Number of Layers
etaInit = 0.1; % Learning Rate Initial Value
Delta = 0.01; % Stop Criterion #1
theta = 0.01; % Stop Criterion #2
epochs = 10000; % Max number of epochs
N = size(X,2); % Number of Input Vectors
nHiddenLayerVec = [3,5,8,11,16,22]; % Number of Hidden Neurons
alpha = 0.999; % Learning Rate Decay Factor
mu = 0.2; % Momentum constant;
M = 10; % Number of Folds (N = leave-one-out)
shuffle = 0; % Shuffle control (turn it off for reproductibility)
smote = 0; % SMOTE oversampling

%% SETTING UP SMOTE
if(smote),
    smoteStr = 'Yes';
else
    smoteStr = 'No';
end;
    
fprintf('----------------------------\n');
fprintf('- Neural Networks Training -\n');
fprintf('----------------------------\n\n');

fprintf('Dataset: %s\n', file);

% Setting SMOTE up
oldInputSize = N;
if(smote),
    data = [X ; t']';

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
    X = filteredData;
    X(size(filteredData, 1),:) = [];
    
    if(M == N),
        N = size(X,2); 
        M = N;
    else
        N = size(X,2);
    end;  
end;

fprintf('Data Balancing? %s\n\n', smoteStr);

[numAttr,numExamples] = size(X);

% Shuffling data
if (shuffle),
    X = [X ; t ; randn(1,numExamples)]';
    X = sortrows(X,numAttr+2)';
    t = X(numAttr+1,:); 
    X = X(1:numAttr,:);
end;

%% BUILDING NEURAL NETWORK AND TRAINING

% Iterate through Different Number of Hidden Layers
for h=1:length(nHiddenLayerVec);
    % Define the Number of Hidden Layer Neurons to use.
    nHiddenLayer = nHiddenLayerVec(h);
    % Create the Empty Matrix to Store Results
    % M-fold cross validation
    answer = zeros(M, floor(N/M));
    target = zeros(M, floor(N/M));
    
    % Store values of the Confusion matrix
    tp = zeros(1, M);
    fp = zeros(1, M);
    tn = zeros(1, M);
    fn = zeros(1, M);
    acc = zeros(1, M);
    timing = zeros(1,M);
    
    % Set the Average Accuracy Value
    avgAcc = 0;
    
    % Print Statements
    fprintf('Training type: %d folds on %d examples\n', M, oldInputSize);
    fprintf('Number of Hidden Layers in Use: %d\n', nHiddenLayer);
    
    % Process Folds
    for fold = 1:M,
        
        % Processing SMOTE
        if(smote && fold <= oldInputSize),
            fprintf('Training fold %d/%d...\n', fold, oldInputSize);
        else
            if(fold <= oldInputSize),
                fprintf('Training fold %d/%d...\n', fold, M);
            end;
        end;
        
        % Layers initialization - for multi-class learning, this should be
        % changed
        L(1).W = (rand(nHiddenLayer,numAttr)-0.5)*0.2; % Weight
        L(1).b = (rand(nHiddenLayer,1)-0.5)*0.2; % Bias
        L(2).W = (rand(1,nHiddenLayer)-0.5)*0.2; % Weight
        L(2).b = (rand(1,1)-0.5)*0.2; % Bias
        
        % Number of Hidden Layers
        for k = 1:K,
            L(k).vb = zeros(size(L(k).b));
            L(k).vW = zeros(size(L(k).W));
        end;
        
        % Sequential Error Backpropagation Training
        n = 1; i = 1; finish = 0; eta = etaInit;
        round = 1; 
        A(fold,round) = 0;
        while not(finish),
            % Start Stopwatch
            foldTime = tic;
            
            % Checking if it is a fold example
            ignoreTraining = 0; ignoreTesting = 0;
            if ((n > ((fold-1)*floor(N/M))) && (n <= (fold*floor(N/M)))),
                ignoreTraining = 1;
                if((n > N) && ~shuffle),
                    ignoreTesting = 1;
                    break;
                end;
            end;
            
            J(fold, i) = 0;
            if(~ignoreTraining);
                for(k = 1:K);
                    L(k).db = zeros(size(L(k).b));
                    L(k).dW = zeros(size(L(k).W));
                end;
            end;
            
            for(ep = n:(n+E-1));
                
                if((ep > N) || ignoreTraining);
                    break;
                end;
                
                % Feed-Forward
                L(1).x = X(:,ep);
                for k = 1:K,
                    L(k).u = L(k).W * L(k).x + L(k).b;
                    L(k).o = tanh(L(k).u); % Activation Function
                    L(k+1).x = L(k).o;
                end;
                e = t(n) - L(K).o;
                J(fold,i) = J(fold,i) + (e'*e)/2;
                
                % Error Backpropagation
                L(K+1).alpha = e;
                L(K+1).W = eye(length(e));
                for k = fliplr(1:K),
                    L(k).M = eye(length(L(k).o)) - diag(L(k).o)^2;
                    L(k).alpha = L(k).M*L(k+1).W'*L(k+1).alpha;
                    L(k).db = L(k).db + L(k).alpha;
                    L(k).dW = L(k).dW + kron(L(k).x',L(k).alpha);
                end;
            end;
            
            % Updates
            for k = 1:K,
                if(ignoreTraining),
                    break;
                end;
                L(k).vb = eta*L(k).db + mu*L(k).vb;
                L(k).b = L(k).b + L(k).vb;
                L(k).vW = eta*L(k).dW + mu*L(k).vW;
                L(k).W = L(k).W + L(k).vW;
            end;
            
            if(~ignoreTraining),
                A(fold,round) = A(fold,round) + (J(fold, i)/(N-1));
                J(fold, i) = J(fold, i)/E;
            end;
            
            % Stop criterion
            if ((i > 1) && (n == N)),
                if (((A(fold,round) < Delta) && ((round > 2) && (abs(A(fold,round-2)-A(fold,round-1) < theta) && (abs(A(fold,round-1)-A(fold,round)) < theta)))) || (i > epochs)),
                    finish = 1;
                end;
            end;
            if not(finish)
                i = i+1; n = n+1;
                if n > N,
                    n = 1;
                    round = round + 1;
                    A(fold,round) = 0;
                end;
                eta = eta*alpha;
            end;
            % Stop Stopwatch and Record Time
            timing(fold) = toc(foldTime);
        end;
        
        % Test
        % fprintf('Testing fold %d/%d...\n', m, M);
        
        if(~ignoreTesting),
            index = 0;
            for n = ((fold-1)*floor(N/M))+1:fold*floor(N/M),
                index = index + 1;
                L(1).x = X(:,n);
                for k = 1:K,
                    L(k).u = L(k).W*L(k).x + L(k).b;
                    L(k).o = tanh(L(k).u);
                    L(k+1).x = L(k).o;
                end;
                answer(fold,index) = L(K).o;
                target(fold,index) = t(n);
                
                if abs(answer(fold,index) - target(fold,index)) < (abs(classes(2) - classes(1))/2),
                    if(abs(answer(fold,index)-classes(1)) < abs(answer(fold,index)-classes(2))), %TP case
                        tp(fold) = tp(fold) + 1; %TP case
                    else
                        tn(fold) = tn(fold) + 1; %TN case
                    end;
                else
                    if(answer(fold,index) > 0), %FP case
                        fp(fold) = fp(fold) + 1;
                    else                     %FN case
                        fn(fold) = fn(fold) + 1;
                    end;
                end;
            end;
        end;
        
        acc(fold) = acc(fold) + ((tp(fold) + tn(fold))/(tp(fold) + tn(fold) + fp(fold) + fn(fold)));
        avgAcc = avgAcc + (acc(fold)/M);
        
    end;
    % Print results
    fprintf('Average accuracy: %f\n', avgAcc);
    fprintf('Average Time to Train, Validate and Test each Fold %f Seconds\nTotal Time to Train %f Seconds\n\n', mean(timing), sum(timing));
end;

%% SCRIPT COMPLETE
fprintf('-----------------------------------\n');
fprintf('- End of Neural Networks Training -\n');
fprintf('-----------------------------------\n\n'); 