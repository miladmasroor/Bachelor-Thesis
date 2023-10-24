 clc;
 clear;
 close all;
 
 load Xnt11;               
 load Ynt11;  
 
 x= Xnt11';
 t =Ynt11';



% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
%trainFcn = 'traingdx';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize =55;
TF={'tansig','purelin'};
net = newff(x,t,hiddenLayerSize,TF);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainlm';
% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
net.trainParam.showWindow=true;
net.trainParam.showCommandLine=false;
net.trainParam.show=1;
net.trainParam.epochs=1000;
net.trainParam.max_fail=20;
net.trainParam.goal=1e-8;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainIdx=find(~isnan(tr.trainMask{1} (1,:)));
valIdx=find(~isnan(tr.valMask{1} (1,:)));
testIdx=find(~isnan(tr.testMask{1} (1,:)));


trainTargets = t(: , trainIdx);
valTargets = t(: , valIdx);
testTargets = t(: ,testIdx);

trainInputs = x(: , trainIdx);
valInputs = x(: , valIdx);
testInputs = x(: ,testIdx);

trainOutputs = y(: , trainIdx);
valOutputs = y(: , valIdx);
testOutputs = y(: ,testIdx);

trainErrors= trainTargets - trainOutputs;
valErrors= valTargets - valOutputs;
testErrors= testTargets - testOutputs;

trainPerformance = perform(net,trainTargets,trainOutputs);
valPerformance = perform(net,valTargets,valOutputs);
testPerformance = perform(net,testTargets,testOutputs);

PlotResults(t,y,'All Data');
PlotResults(trainTargets,trainOutputs,'Train Data');
PlotResults(valTargets,valOutputs,'Validation Data');
PlotResults(testTargets,testOutputs,'Test Data');
% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
figure; 
plotperform(tr);
%figure; 
%plottrainstate(tr);
%figure; 
%ploterrhist(e);
figure;
plotregression(trainTargets , trainOutputs , 'Train Data' , ...
    valTargets , valOutputs , 'Validation Data' , ...
    testTargets , testOutputs , 'Test Data' , ...
    t , y , 'All Data' );
    
%figure; 
%plotfit(net,x,t);

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
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
