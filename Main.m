%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMPIRICAL METHODS FOR FINANCE
% Homework II
%
% Benjamin Souane, Antoine-Michel Alexeev and Julien Bisch
% Due Date: 2 April 2020
%==========================================================================

close all
clc

%Setting the current directory
cd('C:\Users\Benjamin\OneDrive\Documents\GitHub\EMF_HW4');

%import KevinShepperd Toolbox
addpath(genpath('C:\Users\Benjamin\OneDrive\1. HEC\Master\MScF 4.2\EMF\2020\Homeworks\KevinSheperdToolBox'));

%Add the path for the libraries
addpath(genpath('C:\Users\Benjamin\OneDrive\Documents\GitHub\EMF_HW4\Functions'));

%% Importing the data

ImportData;
Price = table2array(DATAHW4(:,2:end));
Date = DATAHW4(:,1);
Names = {'SP 500 Composites','JP USA Gov. Bond','Risk Free'};

clear DATAHW4

%% 1. Computing returns

Returns = PriceToReturn(Price(:,1:2));
Returns_RF = (Price(:,3)/100)/52; % We de-annualized the returns

%% 2. Static asset allocation with constant expected returns and volatility

Lambda = [2,10];
MeanReturns = mean(Returns);
RiskFree = mean(Returns_RF);
CovMat = cov(Returns);
InvCov = inv(CovMat); 

MeanVarWeights = zeros(2,3);

%Computing the allocation
for i = 1:size(Lambda,2)
    MeanVarWeights(i,1:2) = 1/Lambda(i)*InvCov*(MeanReturns' - ones(2,1)*RiskFree);
end

MeanVarWeights(:,3) = 1 - sum(MeanVarWeights,2);

MeanVarAllocation = array2table(MeanVarWeights,'VariableNames',...
{'S&P 500 Comp','JP US Bond', 'Risk Free'},'RowNames', {'Lambda = 2','Lambda = 10'});


%% 3. Estimation of Garch Model

%**************************************************************************
% a. Testing Normality
%**************************************************************************

%Computing Excess Returns
ExcessReturns = Returns - Returns_RF(2:end);

% Lilliefors test
[SP.h,SP.p,SP.ksstat,SP.cv] = lillietest(ExcessReturns(:,1));
[Bond.h,Bond.p,Bond.ksstat,Bond.cv] = lillietest(ExcessReturns(:,2));

% LjungBox test
SP.LjungBox = LjungBoxTest(ExcessReturns(:,1),4,0,0.05);
Bond.LjungBox = LjungBoxTest(ExcessReturns(:,2),4,0,0.05);


% Creating table of the results
Normality = array2table([SP.h,SP.p,SP.ksstat,SP.cv;Bond.h,Bond.p,Bond.ksstat,...
    Bond.cv],'VariableNames',{'Decision','P-Value','K-Stat','Critical Value'}...
    ,'RowNames',{'S&P 500 Comp','JP US Bond'});


AutoCorrelation = array2table([SP.LjungBox;Bond.LjungBox],'VariableNames',...
    {'T-stat','Critical Value','P-Value'},'RowNames',...
    {'S&P 500 Comp','JP US Bond'});


%**************************************************************************
% b. Filter out Correlation with AR(1) model
%**************************************************************************

% Fitting the AR(1)
SP.AR1 = fitlm(Returns(1:end-1,1),Returns(2:end,1));
Bond.AR1 = fitlm(Returns(1:end-1,2),Returns(2:end,2));

% Taking out the Error term
Eps = [table2array(SP.AR1.Residuals(:,1)), table2array(Bond.AR1.Residuals(:,1))];

% Table of AR(1) fitting
AR1 = array2table([SP.AR1.Coefficients.Estimate(1),SP.AR1.Coefficients.tStat(1),...
    SP.AR1.Coefficients.Estimate(2),SP.AR1.Coefficients.tStat(2),...
    SP.AR1.Rsquared.Ordinary;Bond.AR1.Coefficients.Estimate(1),Bond.AR1.Coefficients.tStat(1),...
    Bond.AR1.Coefficients.Estimate(2),Bond.AR1.Coefficients.tStat(2),...
    Bond.AR1.Rsquared.Ordinary],'VariableNames', {'Intercept','tStat','Rho',...
    'tStat_Rho','Rsquared'},'RowNames',{'S&P 500 Comp','JP US Bond'});


%**************************************************************************
% c. Testing ARCH effect on Eps (LM test)
%**************************************************************************

% Fitting a linear model (LM procedure)
SP.ARCH = fitlm([Eps(4:end-1,1).^2,Eps(3:end-2,1).^2 ...
    ,Eps(2:end-3,1).^2, Eps(1:end-4,1).^2],Eps(5:end,1).^2);

Bond.ARCH = fitlm([Eps(4:end-1,2).^2,Eps(3:end-2,2).^2 ...
    ,Eps(2:end-3,2).^2, Eps(1:end-4,2).^2],Eps(5:end,2).^2);


% Extract properties
SP.LM.Stat = (SP.ARCH.NumObservations+4)*SP.ARCH.Rsquared.Ordinary;
SP.LM.Pval = 1 - chi2cdf(SP.LM.Stat,4);
SP.LM.CritVal = chi2inv(0.95,4);
Bond.LM.Stat = (Bond.ARCH.NumObservations+4)*Bond.ARCH.Rsquared.Ordinary;
Bond.LM.Pval = 1 - chi2cdf(Bond.LM.Stat,4);
Bond.LM.CritVal = chi2inv(0.95,4);

% Compare with standard LjungBox on eps^2
SP.LjungBoxSquared = LjungBoxTest(Eps(:,1).^2,4,0,0.05);
Bond.LjungBoxSquared = LjungBoxTest(Eps(:,2).^2,4,0,0.05);

% Create a table of the results
ARCH = array2table([SP.LM.Stat,SP.LM.Pval,SP.LjungBoxSquared(1,1),SP.LjungBoxSquared(1,3);...
    Bond.LM.Stat,Bond.LM.Pval,Bond.LjungBoxSquared(1,1),Bond.LjungBoxSquared(1,3)],...
    'VariableNames',{'LM Stat','P-Value','Q Stat','PValue'},'RowNames',{'S&P 500 Comp','JP US Bond'});


%**************************************************************************
% d. GARCH model using ML 
%**************************************************************************

% Fitting the model
[SP.GARCH.param, SP.GARCH.LL, SP.GARCH.sigmaHat, ~,SP.GARCH.VCV] = tarch(Eps(:,1),1,0,1);
[Bond.GARCH.param, Bond.GARCH.LL, Bond.GARCH.sigmaHat, ~,Bond.GARCH.VCV] = tarch(Eps(:,2),1,0,1);

% Computing t-Stat
SP.GARCH.tStat = zeros(3,1);
Bond.GARCH.tStat = zeros(3,1);

for i = 1:3
      SP.GARCH.tStat(i) = SP.GARCH.param(i)/sqrt(SP.GARCH.VCV(i,i));
      Bond.GARCH.tStat(i) = Bond.GARCH.param(i)/sqrt(Bond.GARCH.VCV(i,i));
end

% Creating a table 
GARCH = array2table([SP.GARCH.param,SP.GARCH.tStat,Bond.GARCH.param,Bond.GARCH.tStat],...
'VariableNames',{'Stock','Stock tStat','Bond','Bond tStat'},'RowNames',{'Omega','Alpha','Beta'});


%**************************************************************************
% e. Volatility Forecasting
%**************************************************************************

NumDays = 52; % Number of days to forecast




