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

% ADD SMOTE IMPLEMENTATION HERE


% Neural Network Parameters
Reg         = [1e-5];                   % Regularisation Value
epochs      = [50 100 200 500];         % Number of Epochs to Train Network with
LR          = [0.1 0.3 0.5 0.7 0.9];    % Learning Rate value
Mo          = [0.5];                    % Momentum
Hidden_N    = [10 20 40 80];            % Number of Hidden Neurons           
K_Folds = 10;                           % Number of K-Folds 
% Define Stopping Criteria.....TO DO

% Find all combinations and 
Comb  = allcomb(Reg, epochs, LR, Mo, Hidden_N);

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
    
    % Completion Percentage
    fprintf('Finding Starting Paramters: %.2f\n', 100*(row_n/length(Ranking_Table)))
end

% Save Ranking Matrix to Text File
csvwrite('Ranking_Matrix.csv',Ranking_Table)

% Creating a Basic Ranked Table
% Rank Each Row by its Cost Function (Error of the Validation Set)
% AND the Accuracy of the Validation Result

% Implement Finer Tuning using an Optimisation Algorithm
% fminsearch - http://uk.mathworks.com/help/matlab/ref/fminsearch.html
