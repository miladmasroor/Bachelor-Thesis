clc;
clear;
close all;

load Ynt11;
load Xnt11;

x=Xnt11';

t=Ynt11';

nData=size(x,2);

perm=randperm(nData);

pTrainData=0.7;
nTrainData=round(pTrainData*nData);
trainInd=perm(1 : nTrainData);
perm(1 : nTrainData)=[];

pTestData=1-pTrainData;
nTestData=nData-nTrainData;
testInd=perm;
% perm(1 :nTestData)=[];

% pValData=1-pTestData-pTrainData;
% nValData=nData-nTestData-nTrainData;
% valInd=perm;

trainTargets = t(: , trainInd);
% valTargets = t(: , valInd);
testTargets = t(: ,testInd);

trainInputs = x(: , trainInd);
% valInputs = x(: , valInd);
testInputs = x(: ,testInd);

% Create and Train RBF Network
Goal=0;
Spread=1;
MaxNeuron=182;
DisplayAt=1;
net=newrb(trainInputs,trainTargets,Goal,Spread,MaxNeuron,DisplayAt);


% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);


trainOutputs = y(: , trainInd);
% valOutputs = y(: , valInd);
testOutputs = y(: ,testInd);

trainErrors= trainTargets - trainOutputs;
% valErrors= valTargets - valOutputs;
testErrors= testTargets - testOutputs;

trainPerformance = perform(net,trainTargets,trainOutputs);
% valPerformance = perform(net,valTargets,valOutputs);
testPerformance = perform(net,testTargets,testOutputs);

PlotResults(t,y,'All Data');
PlotResults(trainTargets,trainOutputs,'Train Data');
% PlotResults(valTargets,valOutputs,'Validation Data');
PlotResults(testTargets,testOutputs,'Test Data');



if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
