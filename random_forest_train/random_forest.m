clc
clear all
close all
warning off

hr_data=readtable('3_18_data.csv');

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
model=fitensemble(dTrain,'is_affect','Bag',100,'Tree','Type','classification');
prediction=predict(model,testing);

%accuracy of prediction (this part I looked up)
maccuracy=(sum(prediction==table2array(dTest(:,end)))/size(dTest,1))*100

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

