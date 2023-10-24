clc;
clear;
close all;

%% Create Time-Series Data
load Xnt11;
load Ynt11;

Inputs=Xnt11;
Targets=Ynt11;

nData=size(Inputs,1);

pTrain=0.7;
nTrainData=round(pTrain*nData);
TrainInputs=Inputs(1:nTrainData,:);
TrainTargets=Targets(1:nTrainData,:);

pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInputs=Inputs(nTrainData+1:end,:);
TestTargets=Targets(nTrainData+1:end,:);

%% Design ANFIS

Option{1}='Grid Part. (genfis1)';
Option{2}='Sub. Clustering (genfis2)';
Option{3}='FCM (genfis3)';

ANSWER=questdlg('Select FIS Generation Approach:',...
                'Select GENFIS',...
                Option{1},Option{2},Option{3},...
                Option{3});
pause(0.1);

switch ANSWER
    case Option{1}
        Prompt={'Number of MFs','Input MF Type:','Output MF Type:'};
        Title='Enter genfis1 parameters';
        DefaultValues={'5','gaussmf','linear'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        nMFs=str2num(PARAMS{1}); %#ok
        InputMF=PARAMS{2};
        OutputMF=PARAMS{3};
        
        fis=genfis1([TrainInputs TrainTargets],nMFs,InputMF,OutputMF);

    case Option{2}
        Prompt={'Influence Radius:'};
        Title='Enter genfis2 parameters';
        DefaultValues={'0.2'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        Radius=str2num(PARAMS{1}); %#ok
        
        fis=genfis2(TrainInputs,TrainTargets,Radius);
        
    case Option{3}
        Prompt={'Number fo Clusters:',...
                'Partition Matrix Exponent:',...
                'Maximum Number of Iterations:',...
                'Minimum Improvemnet:'};
        Title='Enter genfis3 parameters';
        DefaultValues={'10','2','100','1e-5'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        nCluster=str2num(PARAMS{1}); %#ok
        Exponent=str2num(PARAMS{2}); %#ok
        MaxIt=str2num(PARAMS{3}); %#ok
        MinImprovment=str2num(PARAMS{4}); %#ok
        DisplayInfo=1;
        FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
        
        fis=genfis3(TrainInputs,TrainTargets,'sugeno',nCluster,FCMOptions);
end

Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'100','0','0.01','0.9','1.1'};

PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
pause(0.1);

MaxEpoch=str2num(PARAMS{1});                %#ok
ErrorGoal=str2num(PARAMS{2});               %#ok
InitialStepSize=str2num(PARAMS{3});         %#ok
StepSizeDecreaseRate=str2num(PARAMS{4});    %#ok
StepSizeIncreaseRate=str2num(PARAMS{5});    %#ok
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);


%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

figure;
PlotResultss(TrainTargets,TrainOutputs,'Train Data');

figure;
plotregression(TrainTargets,TrainOutputs,'Train Data');
set(gcf,'Toolbar','figure');

%% Apply ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

figure;
PlotResultss(TestTargets,TestOutputs,'Test Data');

figure;
plotregression(TestTargets,TestOutputs,'Test Data');
set(gcf,'Toolbar','figure');

