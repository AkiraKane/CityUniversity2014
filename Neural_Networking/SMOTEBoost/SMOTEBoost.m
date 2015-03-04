function prediction = SMOTEBoost (TRAIN,TEST,WeakLearn,ClassDist)
% This function implements the SMOTEBoost Algorithm. For more details on the 
% theoretical description of the algorithm please refer to the following 
% paper:
% N.V. Chawla, A.Lazarevic, L.O. Hall, K. Bowyer, "SMOTEBoost: Improving 
% Prediction of Minority Class in Boosting, Journal of Knowledge Discovery
% in Databases: PKDD, 2003.
% Input: TRAIN = Training data as matrix
%        TEST = Test data as matrix
%        WeakLearn = String to choose algortihm. Choices are
%                    'svm','tree','knn' and 'logistic'.
%        ClassDist = true or false. true indicates that the class
%                    distribution is maintained while doing weighted 
%                    resampling and before SMOTE is called at each 
%                    iteration. false indicates that the class distribution
%                    is not maintained while resampling.
% Output: prediction = size(TEST,1)x 2 matrix. Col 1 is class labels for 
%                      all instances. Col 2 is probability of the instances 
%                      being classified as positive class.

javaaddpath('weka.jar');

%% Training SMOTEBoost
% Total number of instances in the training set
m = size(TRAIN,1);
POS_DATA = TRAIN(TRAIN(:,end)==1,:);
NEG_DATA = TRAIN(TRAIN(:,end)==0,:);
pos_size = size(POS_DATA,1);
neg_size = size(NEG_DATA,1);

% Reorganize TRAIN by putting all the positive and negative exampels
% together, respectively.
TRAIN = [POS_DATA;NEG_DATA];

% Converting training set into Weka compatible format
CSVtoARFF (TRAIN, 'train', 'train');
train_reader = javaObject('java.io.FileReader', 'train.arff');
train = javaObject('weka.core.Instances', train_reader);
train.setClassIndex(train.numAttributes() - 1);
    
% Total number of iterations of the boosting method
T = 10;

% W stores the weights of the instances in each row for every iteration of
% boosting. Weights for all the instances are initialized by 1/m for the
% first iteration.
W = zeros(1,m);
for i = 1:m
    W(1,i) = 1/m;
end

% L stores pseudo loss values, H stores hypothesis, B stores (1/beta) 
% values that is used as the weight of the % hypothesis while forming the 
% final hypothesis. % All of the following are of length <=T and stores 
% values for every iteration of the boosting process.
L = [];
H = {};
B = [];

% Loop counter
t = 1;

% Keeps counts of the number of times the same boosting iteration have been
% repeated
count = 0;

% Boosting T iterations
while t <= T
    
    % LOG MESSAGE
    disp (['Boosting iteration #' int2str(t)]);
    
    if ClassDist == true
        % Resampling POS_DATA with weights of positive example
        POS_WT = zeros(1,pos_size);
        sum_POS_WT = sum(W(t,1:pos_size));
        for i = 1:pos_size
           POS_WT(i) = W(t,i)/sum_POS_WT ;
        end
        RESAM_POS = POS_DATA(randsample(1:pos_size,pos_size,true,POS_WT),:);

        % Resampling NEG_DATA with weights of positive example
        NEG_WT = zeros(1,neg_size);
        sum_NEG_WT = sum(W(t,pos_size+1:m));
        for i = 1:neg_size
           NEG_WT(i) = W(t,pos_size+i)/sum_NEG_WT ;
        end
        RESAM_NEG = NEG_DATA(randsample(1:neg_size,neg_size,true,NEG_WT),:);
    
        % Resampled TRAIN is stored in RESAMPLED
        RESAMPLED = [RESAM_POS;RESAM_NEG];
        
        % Calulating the percentage of boosting the positive class. 'pert'
        % is used as a parameter of SMOTE
        pert = ((neg_size-pos_size)/pos_size)*100;
    else 
        % Indices of resampled train
        RND_IDX = randsample(1:m,m,true,W(t,:));
        
        % Resampled TRAIN is stored in RESAMPLED
        RESAMPLED = TRAIN(RND_IDX,:);
        
        % Calulating the percentage of boosting the positive class. 'pert'
        % is used as a parameter of SMOTE
        pos_size = sum(RESAMPLED(:,end)==1);
        neg_size = sum(RESAMPLED(:,end)==0);
        pert = ((neg_size-pos_size)/pos_size)*100;
    end
    
    % Converting resample training set into Weka compatible format
    CSVtoARFF (RESAMPLED,'resampled','resampled');
    reader = javaObject('java.io.FileReader','resampled.arff');
    resampled = javaObject('weka.core.Instances',reader);
    resampled.setClassIndex(resampled.numAttributes()-1);
    
    % New SMOTE boosted data gets stored in S
    smote = javaObject('weka.filters.supervised.instance.SMOTE');
    pert = ((neg_size-pos_size)/pos_size)*100;
    smote.setPercentage(pert);
    smote.setInputFormat(resampled);
    
    S = weka.filters.Filter.useFilter(resampled, smote);
    
    % Training a weak learner. 'pred' is the weak hypothesis. However, the 
    % hypothesis function is encoded in 'model'.
    switch WeakLearn
        case 'svm'
            model = javaObject('weka.classifiers.functions.SMO');
        case 'tree'
            model = javaObject('weka.classifiers.trees.J48');
        case 'knn'
            model = javaObject('weka.classifiers.lazy.IBk');
            model.setKNN(5);
        case 'logistic'
            model = javaObject('weka.classifiers.functions.Logistic');
    end
    model.buildClassifier(S);
    
    pred = zeros(m,1);
    for i = 0 : m - 1
        pred(i+1) = model.classifyInstance(train.instance(i));
    end

    % Computing the pseudo loss of hypothesis 'model'
    loss = 0;
    for i = 1:m
        if TRAIN(i,end)==pred(i)
            continue;
        else
            loss = loss + W(t,i);
        end
    end
    
    % If count exceeds a pre-defined threshold (5 in the current
    % implementation), the loop is broken and rolled back to the state
    % where loss > 0.5 was not encountered.
    if count > 5
       L = L(1:t-1);
       H = H(1:t-1);
       B = B(1:t-1);
       disp ('          Too many iterations have loss > 0.5');
       disp ('          Aborting boosting...');
       break;
    end
    
    % If the loss is greater than 1/2, it means that an inverted
    % hypothesis would perform better. In such cases, do not take that
    % hypothesis into consideration and repeat the same iteration. 'count'
    % keeps counts of the number of times the same boosting iteration have
    % been repeated
    if loss > 0.5
        count = count + 1;
        continue;
    else
        count = 1;
    end        
    
    L(t) = loss; % Pseudo-loss at each iteration
    H{t} = model; % Hypothesis function   
    beta = loss/(1-loss); % Setting weight update parameter 'beta'.
    B(t) = log(1/beta); % Weight of the hypothesis
    
    % At the final iteration there is no need to update the weights any
    % further
    if t==T
        break;
    end
    
    % Updating weight    
    for i = 1:m
        if TRAIN(i,end)==pred(i)
            W(t+1,i) = W(t,i)*beta;
        else
            W(t+1,i) = W(t,i);
        end
    end
    
    % Normalizing the weight for the next iteration
    sum_W = sum(W(t+1,:));
    for i = 1:m
        W(t+1,i) = W(t+1,i)/sum_W;
    end
    
    % Incrementing loop counter
    t = t + 1;
end

% The final hypothesis is calculated and tested on the test set
% simulteneously.

%% Testing SMOTEBoost
n = size(TEST,1); % Total number of instances in the test set

CSVtoARFF(TEST,'test','test');
test = 'test.arff';
test_reader = javaObject('java.io.FileReader', test);
test = javaObject('weka.core.Instances', test_reader);
test.setClassIndex(test.numAttributes() - 1);

% Normalizing B
sum_B = sum(B);
for i = 1:size(B,2)
   B(i) = B(i)/sum_B;
end

prediction = zeros(n,2);

for i = 1:n
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_zero = 0;
    wt_one = 0;
    for j = 1:size(H,2)
       p = H{j}.classifyInstance(test.instance(i-1));      
       if p==1
           wt_one = wt_one + B(j);
       else 
           wt_zero = wt_zero + B(j);           
       end
    end
    
    if (wt_one > wt_zero)
        prediction(i,:) = [1 wt_one];
    else
        prediction(i,:) = [0 wt_one];
    end
end