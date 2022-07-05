clc
clear all
close all
warning off

hr_data=readtable('test_hr_data.csv');

%training the model
cv=cvpartition(size(hr_data,1),'HoldOut',0.3);
index=cv.test;
dTrain=hr_data(~index,2:end);
dTest=hr_data(index,2:end);
testing=dTest(1:end,1:end-1);
model=fitensemble(dTrain,'is_affect','Bag',100,'Tree','Type','classification');
prediction=predict(model,testing);

%accuracy of prediction (this part I looked up)
ms=(sum(prediction==table2array(dTest(:,end)))/size(dTest,1))*100

%visualizing classification outcome
a=min(hr_data.bpm):0.1:max(hr_data.bpm);
b=min(hr_data.rr):0.1:max(hr_data.rr);
[x1 x2]=meshgrid(a,b);
x=[x1(:) x2(:)];
ms=predict(model,x);
gscatter(x1(:),x2(:),ms,'cy');
hold on;
gscatter(dTrain.bpm,dTrain.rr,dTrain.is_affect,'rg','.',30);
