function [MSE, model, trn_acc] = train_nn(configuration, Weights, Bias, train_data, training_label, validation_data, validation_label)

    %% Determine the Sizes of the Matrix
    [Instances,Dimensions] = size(train_data);              % Instances = Training Instances, Dimensions = Number of Dimensions
    depth_of_Network = length(configuration.hidNum)+1;      % depth = Number of Hidden Layers + Output Layer 
    labNum = size(unique(training_label),1);                % Number of Different Labels

    %% Setting Up the Weight(s) Matrix
    if isempty(Weights)
        % Initialize Weights
        model.Weights{1} = (1/Dimensions)*(2*rand(Dimensions,configuration.hidNum(1))-1);
        Delta_Weights{1} = zeros(size(model.Weights{1}));
        for i=2:depth_of_Network-1
            model.Weights{i} = (1/configuration.hidNum(i-1))*(2*rand(configuration.hidNum(i-1),configuration.hidNum(i))-1);
            Delta_Weights{i} = zeros(size(model.Weights{i}));
        end
        model.Weights{depth_of_Network} = (1/configuration.hidNum(depth_of_Network-1))*(2*rand(configuration.hidNum(depth_of_Network-1),labNum)-1);
        Delta_Weights{depth_of_Network} = zeros(size(model.Weights{depth_of_Network}));
    end

    %% Setting up the Bias Matrix(s)
    if isempty(Bias)
        % Initialize Bias
        for i=1:depth_of_Network-1
            model.Bias{i} = zeros(1,configuration.hidNum(i));
            Delta_Bias{i} = model.Bias{i};
        end
        model.Bias{depth_of_Network} = zeros(1,labNum);
        Delta_Bias{depth_of_Network} = model.Bias{depth_of_Network};
    end
    
    %% Saving Plot Data
    plot_trn_acc = [];
    plot_vld_acc = [];
    plot_mse = [];
    
    %% Starting the Training
    for e=1:configuration.eNum;
        MSE = 0;
        %% Batch Processing
        for b=1:configuration.bNum;
            %% Dividing the Data up
            inx = (b-1)*configuration.sNum+1:min(b*configuration.sNum,Instances);
            batch_x = train_data(inx,:);
            batch_y = training_label(inx)+1;
            sNum = size(batch_x,1);
            %% Feedforward
            % Forward mesage to get output
            input{1} = bsxfun(@plus,batch_x*model.Weights{1},model.Bias{1});
            actFunc=  str2func(configuration.activationFnc{1});
            output{1} = actFunc(input{1});
            for i=2:depth_of_Network;
                input{i} = bsxfun(@plus,output{i-1}*model.Weights{i},model.Bias{i});
                actFunc=  str2func(configuration.activationFnc{i});
                output{i} = actFunc(input{i});
            end
            
            % Back-prop update
            y = discrete2softmax(batch_y,labNum);
            %disp([y output{depth}]);
                      
            %% Calculate Output Error
            err{depth_of_Network} = (y-output{depth_of_Network}).*deriv(configuration.activationFnc{depth_of_Network},input{depth_of_Network});
            %err
            [~,Output_Val] = max(output{depth_of_Network},[],2);
            %sum(sum(batch_y+1==cout))
            MSE = MSE + mean(sqrt(mean((output{depth_of_Network}-y).^2)));
            
            %% Backpropagation Phase
            for i=depth_of_Network:-1:2;
                diff = output{i-1}'*err{i}/sNum;
                Delta_Weights{i} = configuration.params(1)*(diff - configuration.params(4)*model.Weights{i}) + configuration.params(3)*Delta_Weights{i};
                model.Weights{i} = model.Weights{i} + Delta_Weights{i};

                Delta_Bias{i} = configuration.params(1)*mean(err{i}) + configuration.params(3)*Delta_Bias{i};
                model.Bias{i} = model.Bias{i} + Delta_Bias{i};
                err{i-1} = err{i}*model.Weights{i}'.*deriv(configuration.activationFnc{i},input{i-1});
            end
            %% Whats THIS?
            diff = batch_x'*err{1}/sNum;
            %% Update the Weights in the Network
            Delta_Weights{1} = configuration.params(1)*(diff - configuration.params(4)*model.Weights{1}) + configuration.params(3)*Delta_Weights{1};
            model.Weights{1} = model.Weights{1} + Delta_Weights{1};
            %% Updating the Bias Weights
            Delta_Bias{1} = configuration.params(1)*mean(err{1}) + configuration.params(3)*Delta_Bias{1};
            model.Bias{1} = model.Bias{1} + Delta_Bias{1};
        end
        %% Training and Calculation the Accuracy of the Network - Training Set
        Output_Train = run_nn(configuration.activationFnc, model, train_data);
        trn_acc = sum((Output_Train-1)==training_label)/size(training_label,1);
        %% Training and Calculation the Accuracy of the Network - Validation Set
        Output_Val = run_nn(configuration.activationFnc,model,validation_data);
        vld_acc = sum((Output_Val-1)==validation_label)/size(validation_label,1);
        
        %fprintf('[Eppoch %4d] MSE = %.9f | Train Acc = %.9f | Validation Acc = %.9f\n',e,MSE,trn_acc,vld_acc);
        
        % Collect data for plot
        plot_trn_acc = [plot_trn_acc trn_acc];
        plot_vld_acc = [plot_vld_acc vld_acc];
        plot_mse     = [plot_mse MSE];
        %pause;
    end
    %fig1 = figure(2);
    %set(fig1,'Position',[10,20,300,200]);
    %plot(1:size(plot_trn_acc,2),plot_trn_acc,'r');
    %hold on;
    %plot(1:size(plot_vld_acc,2),plot_vld_acc);
    %legend('Training','Validation');
    %xlabel('Epochs');ylabel('Accuracy');

    %fig2 = figure(3);
    %set(fig2,'Position',[10,20,300,200]);
    %plot(1:size(plot_mse,2),plot_mse);
    %xlabel('Epochs');ylabel('MSE');
end
