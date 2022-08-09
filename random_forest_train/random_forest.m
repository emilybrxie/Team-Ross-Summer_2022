
function maccu = random_forest(mdata, tree_num)
    %creates a random forest model from the given data & number of trees
    %returns the percent accuracy of the model (maccu) & a feature
    %importance plot
    %   inputs:
    %       mdata: A [m-by-n] matrix, where n is the number of variables
    %       present in the data (n>=3).
    %       In addition, the columns must be in the following order:
    %           time  bpm  rr  ......  is_affect 
    %       (i.e. time must be 1st col & is_affect must be last col)
    %
    %       tree_num: an integer that represents the number of trees the
    %       model needs to contain when training. If false, then it is set
    %       to default (100 trees)
    %   outputs:
    %       maccu: model accuracy, in percentage
    %       an additional feature importance plot


    %determine num of trees
    if tree_num
        numTrees=tree_num;
    else
        numTrees=100;
    end

    %convert matrix to table
    hr_data=array2table(mdata);
    hr_data.Properties.VariableNames(1:8) = {'time','bpm','rr','rmssd','sdsd','sdnn','pnnx','is_affect'};


    %fill in NaNs in time & 0 in bpm
    rows=size(hr_data,1);
    for row = 1:rows
        A=hr_data{row,:};
        TF=isnan(A);
        if TF(1)==true
            hr_data(row,1)=hr_data(row-1,1);
            hr_data(row,2)=hr_data(row-1,2);
        end
    end


    %training the model
    cv=cvpartition(size(hr_data,1),'HoldOut',0.2);
    index=cv.test;
    dTrain=hr_data(~index,2:end);
    dTest=hr_data(index,2:end);
    testing=dTest(1:end,1:end-1);
    model=fitensemble(dTrain,'is_affect','Bag',numTrees,'Tree','Type','classification');
    prediction=predict(model,testing);

    %accuracy of prediction (this part I looked up)
    maccu=(sum(prediction==table2array(dTest(:,end)))/size(dTest,1))*100

    %feature importance
    feature_importance=oobPermutedPredictorImportance(model);
    figure
    bar(feature_importance)
    title('Feature Importance Estimates')
    xlabel('Predictor variable')
    ylabel('Importance')
    h=gca;
    h.XTickLabel=model.PredictorNames;
    h.XTickLabelRotation=45;
    h.TickLabelInterpreter='none';

end