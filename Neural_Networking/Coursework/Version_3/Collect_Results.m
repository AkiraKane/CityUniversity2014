function Collect_Results()

% WORK OUTSTANDING - Difficulty - Estimated Delivery Date
%   CROSS VALIDATION - Easy - By Monday - COMPLETE
%   SAVING JPEG FILES OF CONFUSION MATRIX - Easy - By Monday
%   RANKING ALGORITHM - Medium - By Tuesday - COMPLETE including Saving of Results table
%   OPTIMISATION USING FMINSEARCH - Medium - By Tuesday
%   IMPORT BANKING DATA - Easy - By Monday
%   ADD ABILITY TO ADD MULTIPLE HIDDEN LAYERS - Hard - By Wednesday - COMPLETED
%   AMEND PYTHON SCRIPT TO CONVERT DATA - Easy - By Sunday
%   ADD THE ABILITY TO USE DIFFERENT ACTIVATION FUNCs - Medium - By Tuesday
%   ADD SMOTE Functionality - Medium - By Tuesday

% Import Data - TO DO - Import Data

% X = Training Data
Full = [1 0;0 1;1 1;0 0];

% X_target = Training Data Labels
Full_target = [1 1 0 0]';

% Neural Network Parameters
Reg     = [1e-5 1e-3 0.01];     % Regularisation Value
epochs  = [3000 1500];      % Number of Epochs to Train Network with
LR      = [0.9 0.5 0.1 0.01];   % Learning Rate value
Mo      = [0.9 0.5 0.1];        % Momentum
K_Folds = 4;                   % Number of K-Folds 
% Define Stopping Criteria.....TO DO

% Find all combinations and 
Comb  = allcomb(Reg, epochs, LR, Mo);

% Results Table for Ranking - 6 Columns
%    Store Cost Of Validation
%    Accuracy of Validation
%    Regularisation
%    Epochs
%    Learning Rate
%    Momentum
%    K - Fold Data - TO ADD
Ranking_Table = zeros(size(Comb, 1), 6); 

% Loop Through all Combinations
for row_n = 1:10    % 50 % length(Ranking_Table) % Add this in to run throgh all combinations
    
    % Get Indices for Cross Validation
    Indices = crossvalind('Kfold', length(Full), K_Folds);    
    
    % Save Fold Results to a Temporary Table for Aggregating
    CV_Results = zeros(K_Folds, 10);
    
    for fold = 1:K_Folds; % Loop through CV Loops
        
         % TEMPORARY SOLUTION AS THE NUMBER OF EXAMPLEs IS LOW
         Y = Full; X = Full;
         X_target = Full_target; Y_target = Full_target;
          
%         % Slice and Dice the Datasets
%         Y = Full(Indices == fold, 1:size(Full,2));         % Create the Validation Data
%         Y_target = Full_target(Indices == fold,1);         % Create the Validation Labels
%         X = Full(Indices ~=fold, 1:size(Full,2));             % Create the Training Data
%         X_target = Full_target(Indices ~=fold, 1);         % Create the Training Labels
        
        % Initiate, Train and Score Models
        CV_Results(fold, :) = Create_And_Train(X, X_target, Y, Y_target, Comb(row_n,2), Comb(row_n,3), Comb(row_n,4), Comb(row_n,1), 0);
        
    end
    % Aggregate Folds Data - Saving...
    %   Accuracy of Validation
    %   Cost Functions of Validation
    %   + Initial Setup
    Ranking_Table(row_n,:) = [mean(CV_Results(:, 6)) mean(CV_Results(:, 10)) Comb(row_n,1), Comb(row_n,2), Comb(row_n,3), Comb(row_n,4)];
    
    % Completion Percentage
    fprintf('Finding Starting Paramters: %.2f\n', 100*(row_n/length(Ranking_Table)))
end

% Save Ranking Matrix to Text File
%csvwrite('Ranking_Matrix.csv',Ranking_Table)

% Creating a Basic Ranked Table
% Rank Each Row by its Cost Function (Error of the Validation Set)
% AND the Accuracy of the Validation Result

% [Index, Rank Accuracy, Rank Cost Function, Score]
Calculated_Rank = [ [1:size(Ranking_Table, 1)]' tiedrank(Ranking_Table(:,1)) tiedrank(Ranking_Table(:,2)) [tiedrank(Ranking_Table(:,1))+tiedrank(Ranking_Table(:,2))] ];

% Sort Matrix and Get Top 5 Results
[d1,d2] = sort(Calculated_Rank(:,4));
Calculated_Rank = Calculated_Rank(d2,:);

% Get a List of the Top 5 Configurations
Top_5_Initial_Configurations = Ranking_Table(Calculated_Rank(1:5,1),:);
%csvwrite('Top_5_Initial_Configurations.csv', Top_5_Initial_Configurations);

% Implement Finer Tuning using an Optimisation Algorithm
% fminsearch - http://uk.mathworks.com/help/matlab/ref/fminsearch.html
