function Collect_Results()
% Import Data
X = [1 0;0 1;1 1;0 0];
Y=X;

X_target = [1 1 0 0]';
Y_target = X_target;

% Neural Network Parameters
Reg     = [1e-5 1e-3 0.01];     % Regularisation Value
epochs  = [3000 1500 500];   % Number of Epochs to Train Network with
LR      = [0.9 0.5 0.1 0.01];   % Learning Rate value
Mo      = [0.9 0.5 0.1];        % Momentum
%K_Folds = [1:10];               % Number of K-Folds 
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

% Loop Through Combinations
for row_n = 1:length(Ranking_Table)
    % TO ADD CROSS VALIDATION HERE
    Result_Line = Create_And_Train(X, X_target, Y, Y_target, Comb(row_n,2), Comb(row_n,3), Comb(row_n,4), Comb(row_n,1), 0);
    % Aggregate Folds Data
    Ranking_Table(row_n,:) = [Result_Line(:,6) Result_Line(:,10) Comb(row_n,1), Comb(row_n,2), Comb(row_n,3), Comb(row_n,4)];
end

    