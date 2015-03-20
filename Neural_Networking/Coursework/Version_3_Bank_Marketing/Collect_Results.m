function Collect_Results()

% WORK OUTSTANDING - Difficulty - Estimated Delivery Date
%   CROSS VALIDATION - Easy - By Monday - COMPLETE
%   SAVING JPEG FILES OF CONFUSION MATRIX - Easy - By Monday
%   RANKING ALGORITHM - Medium - By Tuesday - COMPLETE including Saving of Results table
%   OPTIMISATION USING FMINSEARCH - Medium - By Tuesday
%   IMPORT BANKING DATA - Easy - By Monday - COMPLETE
%   ADD ABILITY TO ADD MULTIPLE HIDDEN LAYERS - Hard - By Wednesday - COMPLETED
%   AMEND PYTHON SCRIPT TO CONVERT DATA - Easy - By Sunday - COMPLETED
%   ADD THE ABILITY TO USE DIFFERENT ACTIVATION FUNCs - Medium - By Tuesday
%   ADD SMOTE Functionality - Medium - By Tuesday

% Import Data

file = 'Training';                                        % Filename to Import
X = csvread([file '.csv'],1,0);                           % Import the File into an Matrix
Full = X(:,1:63);
Full_target = X(:,64);

% Implementing Smote (synthetic minority over sampling technique)
fprintf('\nImported Dataset Overview by Target Class\n')
Pre_Ammendment = tabulate(Full_target);
tabulate(Full_target)

% Current Directory
folder = cd;

% Adding necessary files to the java path
javaaddpath([folder '/Java_Extensions/weka.jar']);
javaaddpath([folder '/Java_Extensions/smote.jar']);

smote = 1; % SMOTE oversampling ELSE WILL BE USING AN RBM

% Setting SMOTE/RBM up
if(smote),
    data = [Full , Full_target];
    
    %data2arff(data, file);
    
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
    
    % Get the Target Data
    Full_target = (filteredData(size(filteredData, 1),:)*2)-1;
    Full_target = Full_target'; 
    Full_target( Full_target < 0 )= 0;
    fprintf('\nPost SMOTE Result\n')
    tabulate(Full_target)
    Post_Ammendment = tabulate(Full_target);
    
    % Get the Input Data
    Full = filteredData;
    Full(size(filteredData, 1),:) = [];
    Full = Full';
    
    % Display Change in Class Sizes
    fprintf('Change in Class Sizes: %i \n', (Post_Ammendment(:,2) - Pre_Ammendment(:,2)))
else
    fprintf('Generating synthetic Data using a Restricted Boltzmann Machine\n')
end;


% Neural Network Parameters
Reg         = [1e-5 0.1];                   % Regularisation Value
epochs      = [100 150];         % Number of Epochs to Train Network with
LR          = [0.1 0.3 0.5];    % Learning Rate value
Mo          = [0.1];                    % Momentum
Hidden_N    = [20 40 80];            % Number of Hidden Neurons           
K_Folds = 10;                           % Number of K-Folds 
% Define Stopping Criteria.....TO DO

% Find all combinations and 
Comb  = allcomb(Reg, epochs, LR, Mo, Hidden_N)

% Results Table for Ranking - 6 Columns
%    Store Cost Of Validation
%    Accuracy of Validation
%    Regularisation
%    Epochs
%    Learning Rate
%    Momentum
%    No. Hidden Neurons
Ranking_Table = zeros(size(Comb, 1), 9); 

% Get Indices for Cross Validation
% Indc creation is outside of the main Loops to ensure that there alleast
% the data is used for Training and Validation is consistant accross each
% model
Indices = crossvalind('Kfold', length(Full), K_Folds);

% Loop Through all Combinations
for row_n = 1:size(Ranking_Table, 1)    % 50 % length(Ranking_Table) % Add this in to run throgh all combinations
        
    % Save Fold Results to a Temporary Table for Aggregating
    CV_Results = zeros(K_Folds, 10);
    
    for fold = 1:K_Folds; % Loop through CV Loops
                  
        % Slice and Dice the Datasets
        Y = Full(Indices == fold, 1:size(Full,2));         % Create the Validation Data
        Y_target = Full_target(Indices == fold,1);         % Create the Validation Labels
        X = Full(Indices ~=fold, 1:size(Full,2));          % Create the Training Data
        X_target = Full_target(Indices ~=fold, 1);         % Create the Training Labels
        
        % Initiate, Train and Score Models
        CV_Results(fold, :) = Create_And_Train(X, X_target, Y, Y_target, Comb(row_n,2), Comb(row_n,3), Comb(row_n,4), Comb(row_n,1), Comb(row_n, 5), 0);
        
    end
    % Aggregate Folds Data - Saving...
    %   Accuracy of Validation
    %   Cost Functions of Validation
    %   + Initial Setup
    Ranking_Table(row_n,:) = [mean(CV_Results(:, 6)) std(CV_Results(:, 6)) mean(CV_Results(:, 10)) std(CV_Results(:, 10)) Comb(row_n,1) Comb(row_n,2) Comb(row_n,3) Comb(row_n,4) Comb(row_n,5)];
    
    % Save Ranking Matrix to Text File
    csvwrite('Ranking_Matrix.csv',Ranking_Table)

    % Completion Percentage
    fprintf('Finding Starting Paramters: %.2f\n', 100*(row_n/length(Ranking_Table)))
end


% Creating a Basic Ranked Table
% Rank Each Row by its Cost Function (Error of the Validation Set)
% AND the Accuracy of the Validation Result

% Implement Finer Tuning using an Optimisation Algorithm
% fminsearch - http://uk.mathworks.com/help/matlab/ref/fminsearch.html
