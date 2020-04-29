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

% Compute unconditional Variance
SP.Forecast.UncondVar = SP.GARCH.param(1)/(1-SP.GARCH.param(2) - SP.GARCH.param(3));
Bond.Forecast.UncondVar = Bond.GARCH.param(1)/(1-Bond.GARCH.param(2) - Bond.GARCH.param(3));

% Forecast value
SP.Forecast.Var = zeros(NumDays,1);
Bond.Forecast.Var = zeros(NumDays,1);

for i = 1:NumDays
    if i == 1
    
        SP.Forecast.Var(i) = SP.GARCH.param(1) + SP.GARCH.param(2)*Eps(end,1)^2+...
            SP.GARCH.param(3)*SP.GARCH.sigmaHat(end);
        Bond.Forecast.Var(i) = Bond.GARCH.param(1) + Bond.GARCH.param(2)*Eps(end,2)^2+...
            Bond.GARCH.param(3)*Bond.GARCH.sigmaHat(end);        
    else 
        
        SP.Forecast.Var(i) = SP.GARCH.param(1) + (SP.GARCH.param(2) + SP.GARCH.param(3))*...
            SP.Forecast.Var(i-1);
        Bond.Forecast.Var(i) = Bond.GARCH.param(1) + (Bond.GARCH.param(2) + Bond.GARCH.param(3))*...
            Bond.Forecast.Var(i-1);
    end
end 


% Ploting the results
f = figure('Visible','off');
x0 = 10;
y0 = 10;
width = 1000;
height = 400;
set (f, 'position' , [x0, y0, width, height])
subplot(1,2,1)
plot(sqrt(SP.Forecast.Var)*100)
hold on 
plot(sqrt(SP.Forecast.UncondVar)*100*ones(NumDays,1))
xlabel('Number of weeks forecasted')
ylabel('Weekly forecasted volatility in %')
legend('Conditional Volatility','Unconditional Volatility','location','best')
xlim([1 NumDays])
title('SP 500 Weekly Forecasted Vol.')
subplot(1,2,2)
plot(sqrt(Bond.Forecast.Var)*100)
hold on 
plot(sqrt(Bond.Forecast.UncondVar)*100*ones(NumDays,1))
xlabel('Number of weeks forecasted')
ylabel('Weekly forecasted volatility in %')
legend('Conditional Volatility','Unconditional Volatility','location','best')
xlim([1 NumDays])
title('JP Us Bond Weekly Forecasted Vol.')
print(f,'Output/VolatilityForecast','-dpng','-r1000')
clear f


%% 4. Dynamic Asset Allocation

%**************************************************************************
% a. Dynamic Covariance computation 
%**************************************************************************

% Mean Returns using AR(1) Process
SP.Mean = table2array(SP.AR1.Coefficients(1,1)) + table2array(SP.AR1.Coefficients(2,1))*Returns(1:end-1,1);
Bond.Mean = table2array(Bond.AR1.Coefficients(1,1)) + table2array(Bond.AR1.Coefficients(2,1))*Returns(1:end-1,2);

% Covariance Using GARCH(1,1)
Covariance.Rho = corr(Eps(:,1),Eps(:,2));
Covariance.Corr = Covariance.Rho*sqrt(SP.GARCH.sigmaHat).*sqrt(Bond.GARCH.sigmaHat);
Covariance.StockVar = SP.GARCH.sigmaHat;
Covariance.BondVar = Bond.GARCH.sigmaHat;

% Dynamic Volatility Plot
f = figure('Visible','off');
x0 = 10;
y0 = 10;
width = 800;
height = 400;
set (f, 'position' , [x0, y0, width, height])
plot(table2array(Date(3:end,'Name')),sqrt(Covariance.StockVar)*sqrt(52))
hold on 
plot(table2array(Date(3:end,'Name')),sqrt(Covariance.BondVar)*sqrt(52))
xlabel('Date')
ylabel('Annualized Volatility')
legend('SP 500','JP US Bond','location','best')
title('Dynamic Volatility of bonds and stocks')
print(f,'Output/VolatilityDynamic','-dpng','-r1000')
clear f

% Computing the a T x 2 x 2 matrix of Covariance matrix
NumBalancing = size(Covariance.StockVar,1);
Covariance.CovMat = zeros(2,2,NumBalancing);

for i = 1:NumBalancing
    Covariance.CovMat(:,:,i) = [Covariance.StockVar(i),Covariance.Corr(i);...
        Covariance.Corr(i),Covariance.BondVar(i)];
end

%**************************************************************************
% b. Dynamic Asset Allocation 
%**************************************************************************

% Dynamic Expeced Returns
DynamicAllocation.mean = [SP.Mean,Bond.Mean];

% Dynamic Allocation
DynamicAllocation.Lambda2.alpha = zeros(2,NumBalancing);
DynamicAllocation.Lambda2.RF = zeros(1,NumBalancing);
DynamicAllocation.Lambda10.alpha = zeros(2,NumBalancing);
DynamicAllocation.Lambda10.RF = zeros(1,NumBalancing);

for i = 1:NumBalancing
       DynamicAllocation.Lambda2.alpha(:,i) = 1/Lambda(1)*inv(Covariance.CovMat(:,:,i))*...
           (DynamicAllocation.mean(i,:)'-ones(2,1)*Returns_RF(i+2)); 
       DynamicAllocation.Lambda2.RF(i) =  1 - ones(1,2)*DynamicAllocation.Lambda2.alpha(:,i);
       DynamicAllocation.Lambda10.alpha(:,i) = 1/Lambda(2)*inv(Covariance.CovMat(:,:,i))*...
           (DynamicAllocation.mean(i,:)'-ones(2,1)*Returns_RF(i+2)); 
       DynamicAllocation.Lambda10.RF(i) =  1 - ones(1,2)*DynamicAllocation.Lambda10.alpha(:,i);
end


%**************************************************************************
% c. Dynamic Asset Allocation Plot
%**************************************************************************

f = figure('Visible','off');
x0 = 10;
y0 = 10;
width = 1000;
height = 500;
set (f, 'position' , [x0, y0, width, height])
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda2.alpha(1,:))
hold on 
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda2.alpha(2,:))
hold on
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda2.RF)
hold on
plot(xlim,[MeanVarWeights(1,1) MeanVarWeights(1,1)])
hold on 
plot(xlim,[MeanVarWeights(1,2) MeanVarWeights(1,2)])
hold on 
plot(xlim,[MeanVarWeights(1,3) MeanVarWeights(1,3)])
xlabel('Date')
ylabel('Weights Allocation ( 1 = 100%)')
legend('SP 500 Dynamic','JP US Bond Dynamic','Risk Free Dynamic',...
    'SP 500 Static','JP US Bond Static','Risk Free Static','location','bestoutside','Orientation','Horizontal')
title('Dynamic Allocation for Lambda = 2')
print(f,'Output/DynamicAllocation_Lambda2','-dpng','-r1000')
clear f
f = figure('Visible','off');
set (f, 'position' , [x0, y0, width, height])
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda10.alpha(1,:))
hold on 
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda10.alpha(2,:))
hold on
plot(table2array(Date(3:end,'Name')),DynamicAllocation.Lambda10.RF)
hold on
plot(xlim,[MeanVarWeights(2,1) MeanVarWeights(2,1)])
hold on 
plot(xlim,[MeanVarWeights(2,2) MeanVarWeights(2,2)])
hold on 
plot(xlim,[MeanVarWeights(2,3) MeanVarWeights(2,3)])
xlabel('Date')
ylabel('Weights Allocation ( 1 = 100%)')
legend('SP 500 Dynamic','JP US Bond Dynamic','Risk Free Dynamic',...
    'SP 500 Static','JP US Bond Static','Risk Free Static','location','bestoutside','Orientation','Horizontal')
title('Dynamic Allocation for Lambda = 10')
print(f,'Output/DynamicAllocation_Lambda10','-dpng','-r1000')
clear f

%**************************************************************************
% D. Cumulative Returns 
%**************************************************************************

Portfolio.Dynamic.Returns2 = zeros(NumBalancing,1);
Portfolio.Dynamic.Returns10 = zeros(NumBalancing,1);
Portfolio.Static.Returns2 = zeros(NumBalancing,1);
Portfolio.Static.Returns10 = zeros(NumBalancing,1);

LogReturns = log(1 + Returns);
LogReturns_RF = log(1 + Returns_RF);

for i = 1:NumBalancing
    
    Portfolio.Dynamic.Returns2(i) = DynamicAllocation.Lambda2.alpha(:,i)'*Returns(i+1,:)' ...
        + DynamicAllocation.Lambda2.RF(i)*Returns_RF(i+2);

    Portfolio.Dynamic.Returns10(i) = DynamicAllocation.Lambda10.alpha(:,i)'*Returns(i+1,:)' ...
        + DynamicAllocation.Lambda10.RF(i)*Returns_RF(i+2);
    
end

 Portfolio.Static.Returns2 = (MeanVarWeights(1,1:2)*Returns(2:end,:)')' ...
       + MeanVarWeights(1,3).*Returns_RF(3:end);
    
    Portfolio.Static.Returns10 = (MeanVarWeights(2,1:2)*Returns(2:end,:)')' ...
       + MeanVarWeights(2,3).*Returns_RF(3:end);
   
   
Portfolio.Dynamic.CumReturn2 = cumsum(Portfolio.Dynamic.Returns2);
Portfolio.Dynamic.CumReturn10 = cumsum(Portfolio.Dynamic.Returns10);
Portfolio.Static.CumReturn2 = cumsum(Portfolio.Static.Returns2);
Portfolio.Static.CumReturn10 = cumsum(Portfolio.Static.Returns10);

f = figure('Visible','off');
plot(table2array(Date(3:end,'Name')),Portfolio.Dynamic.CumReturn2)
hold on 
plot(table2array(Date(3:end,'Name')),Portfolio.Dynamic.CumReturn10)
hold on
plot(table2array(Date(3:end,'Name')),Portfolio.Static.CumReturn2)
hold on
plot(table2array(Date(3:end,'Name')),Portfolio.Static.CumReturn10)
xlabel('Date')
ylabel('Cumulative Log-Returns')
legend('Dynamic Allocation - Lambda = 2','Dynamic Allocation - Lambda = 10','Static Allocation - Lambda = 2',...
    'Static Allocation - Lambda = 10','location','best')
title('Cumulative Log-Returns of different allocations')
print(f,'Output/CumulativeReturns','-dpng','-r1000')
clear f


% Statistics
Portfolio.All = [Portfolio.Dynamic.Returns2, Portfolio.Dynamic.Returns10, ...
   Portfolio.Static.Returns2, Portfolio.Static.Returns10 ];

Portfolio.Statistics.mean = mean(Portfolio.All)*52;
Portfolio.Statistics.std = std(Portfolio.All)*sqrt(52);
Portfolio.Statistics.sharpe = Portfolio.Statistics.mean./Portfolio.Statistics.std;
Portfolio.Statistics.skewness = skewness(Portfolio.All);
Portfolio.Statistics.kurtosis = kurtosis(Portfolio.All)-3;

PortfolioStatistics = array2table([Portfolio.Statistics.mean;Portfolio.Statistics.std;...
    Portfolio.Statistics.sharpe;Portfolio.Statistics.skewness;Portfolio.Statistics.kurtosis],...
    'VariableNames',{'Dynamic Lambda = 2','Dynamic Lambda = 10','Static Lambda = 2','Static Lambda = 10'},...
    'RowNames',{'Annualized Mean','Annualized Volatility','Sharpe Ratio','Skewness','Kurtosis'});

%**************************************************************************
% E. Fees
%**************************************************************************

% Pre-allocating the output
Portfolio.Fees.Lambda2 = zeros(1,NumBalancing);
Portfolio.Fees.Lambda10 = zeros(1,NumBalancing);

% Computing Fees
Portfolio.Fees.Lambda2(1) = abs(DynamicAllocation.Lambda2.alpha(1,1)) + ...
    abs(DynamicAllocation.Lambda2.alpha(2,1));
Portfolio.Fees.Lambda2(2:end) = abs(DynamicAllocation.Lambda2.alpha(1,2:end)-DynamicAllocation.Lambda2.alpha(1,1:end-1)) + ...
    abs(DynamicAllocation.Lambda2.alpha(2,2:end)-DynamicAllocation.Lambda2.alpha(2,1:end-1));
Portfolio.Fees.Lambda10(1) = abs(DynamicAllocation.Lambda10.alpha(1,1)) + ...
    abs(DynamicAllocation.Lambda10.alpha(2,1));
Portfolio.Fees.Lambda10(2:end) = abs(DynamicAllocation.Lambda10.alpha(1,2:end)-DynamicAllocation.Lambda10.alpha(1,1:end-1)) + ...
    abs(DynamicAllocation.Lambda10.alpha(2,2:end)-DynamicAllocation.Lambda10.alpha(2,1:end-1));

% Computing 'Optimal' Fees
Portfolio.Fees.T1 = (Portfolio.Dynamic.CumReturn2(end) - Portfolio.Static.CumReturn2(end))/sum(Portfolio.Fees.Lambda2);
Portfolio.Fees.T2 = (Portfolio.Dynamic.CumReturn10(end) - Portfolio.Static.CumReturn10(end))/sum(Portfolio.Fees.Lambda10);

% Computing returns with fees
Portfolio.Fees.Returns2 = Portfolio.Dynamic.Returns2 - Portfolio.Fees.Lambda2'*Portfolio.Fees.T1;
Portfolio.Fees.Returns10 = Portfolio.Dynamic.Returns10 - Portfolio.Fees.Lambda10'*Portfolio.Fees.T2;

% Plotting the results
f = figure('Visible','off');
plot(table2array(Date(3:end,'Name')),cumsum(Portofolio.Fees.Returns2))
hold on 
plot(table2array(Date(3:end,'Name')),cumsum(Portofolio.Fees.Returns10))
hold on
plot(table2array(Date(3:end,'Name')),Portfolio.Static.CumReturn2)
hold on
plot(table2array(Date(3:end,'Name')),Portfolio.Static.CumReturn10)
xlabel('Date')
ylabel('Cumulative Log-Returns')
legend('Dynamic Allocation - Lambda = 2','Dynamic Allocation - Lambda = 10','Static Allocation - Lambda = 2',...
    'Static Allocation - Lambda = 10','location','best')
title('After Fees cumulative Log-Returns of different allocations')
print(f,'Output/CumulativeReturnsFees','-dpng','-r1000')
clear f

% Exporting the table in latex 

table2latex(AR1,'Output/LatexCode/AR1');
table2latex(ARCH,'Output/LatexCode/ARCH');
table2latex(GARCH,'Output/LatexCode/GARCH');
table2latex(AutoCorrelation,'Output/LatexCode/AutoCorrelation');
table2latex(MeanVarAllocation,'Output/LatexCode/MeanVarAllocation');
table2latex(Normality,'Output/LatexCode/Normality');
table2latex(PortfolioStatistics,'Output/LatexCode/Port');

% Clearing useless variables

clear width x0 y0 height i
