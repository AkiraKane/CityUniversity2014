function show_setup(net)
    % Function to Display configuration of the NN
    info = length(net.Weights);
    
    % Print to Command Line
    fprintf('\n\nNeural Net Configuration\n\n')
    fprintf('Total No. of Layers: %d No. Hidden Layers: %d \n', info+1, info-1)
    fprintf('Number of Input Dimensions %d\n', size(net.Weights{1},2)-1)
    fprintf('Number of Output Dimensions %d \n\n', size(net.Weights{info},1))
    
    % Print Number of Neurons at each Layer of Net
    for i = 1:info+1
       if i == 1
           fprintf('No.Input Neurons %d\n', size(net.Weights{1},2)-1)
       elseif i-1 < info
           fprintf('No.Hidden Neurons %d\n',size(net.Weights{i-1},1))
       else
           fprintf('No.Output Neurons %d\n',size(net.Weights{i-1},1))
       end
    end