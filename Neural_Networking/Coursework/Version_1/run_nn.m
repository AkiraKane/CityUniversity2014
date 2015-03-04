function cout = run_nn(actFncs,model,dat)
    for i=1:length(actFncs);
        input = bsxfun(@plus,dat*model.Weights{i},model.Bias{i});
        actFunc=  str2func(actFncs{i});
        dat = actFunc(input);
    end      
    [~,cout] = max(dat,[],2);
end

