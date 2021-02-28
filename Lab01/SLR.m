% Course: ESE 588, Spring 2021
% Date Created: 2/27/21
% Date Modified: 2/27/21
% Lab 1 - Parameter Estimation for the SLR Model
% SLR - Simple Linear Regression
% Author: Albert Thomas
% Description: Perform SLR with the Fire Damage Data Set and estimate 
% parameters. 
% Parameters: Beta 1 hat, Beta 2 hat, y hat, residuals, variance of Beta 1 
% hat, variance of Beta 2 hat, estimation of noise variance (omega squared)

% set up FireDamage Data---------------------------------------
elements = 15;
FireDamage = zeros(elements,2);
FireDamage(1,:) = [3.4, 26.2];
FireDamage(2,:) = [1.8, 17.7];
FireDamage(3,:) = [4.6, 31.3];
FireDamage(4,:) = [2.3, 23.1];
FireDamage(5,:) = [3.1, 27.5];
FireDamage(6,:) = [5.5, 36.0];
FireDamage(7,:) = [.7, 14.1];
FireDamage(8,:) = [3.0, 22.3];
FireDamage(9,:) = [2.6, 19.6];
FireDamage(10,:) = [4.3, 31.3];
FireDamage(11,:) = [2.1, 24.0];
FireDamage(12,:) = [1.1, 17.3];
FireDamage(13,:) = [6.1, 43.2];
FireDamage(14,:) = [4.8, 36.4];
FireDamage(15,:) = [3.8, 26.1];
% Data has been set up--------------------------------------------
% Calculate xbar and ybar-----------------------------------------
xbar = 0;
ybar = 0;
for i = 1 : elements
    xbar = xbar + FireDamage(i, 1);
    ybar = ybar + FireDamage(i,2);
end
xbar = xbar/elements;
ybar = ybar/elements;
% Calculate xbar and ybar Complete---------------------------------
% Calculate Sxy and Sxx -------------------------------------------
Sxx = 0;
Sxy = 0;
for i = 1:elements
    Sxx = Sxx + (FireDamage(i,1)*FireDamage(i,1));
    Sxy = Sxy + (FireDamage(i,1)*FireDamage(i,2));
end
Sxx = Sxx - ((elements)*(xbar*xbar));%Final Sxx value
Sxy = Sxy - ((elements)*(xbar*ybar));%Final Sxy value

% Calculate Sxy and Sxx Complete ----------------------------------
% Calculate Beta1Hat and Beta0Hat ---------------------------------
Beta1Hat = Sxy/Sxx;
% Beta 0 hat = (y bar) - (Beta 1 hat) (x bar)
Beta0Hat = ybar - (Beta1Hat*(xbar));
% Calculate Beta1Hat and Beta0Hat Complete ------------------------

% Calculate residuals ---------------------------------------------
e = zeros(1,elements);
yhat = zeros(1,elements);
for i = 1:elements
    yhat(i) = Beta0Hat + (Beta1Hat*FireDamage(i,1)); %estimate of y at a given x value
    e(i) = FireDamage(i,2) - yhat(i);
end
% Calculate residuals Complete ------------------------------------

% Plot Scatterplot & Line of Best Fit on Same Graph ---------------
figure(1)
xplot = 0:elements;
yplot = Beta0Hat + (Beta1Hat*xplot);
plot(xplot, yplot);
hold on
scatter(FireDamage(:,1), FireDamage(:,2));
% Sum of Squared Errors (SSE) Calculation
SSE = 0;
for i = 1:elements
    SSE = SSE + (e(i) * e(i));
end

NoiseVariance = SSE/(elements-2);
VarianceB1H = (NoiseVariance*NoiseVariance)/(Sxx);
VarianceB0H = (NoiseVariance*NoiseVariance)*((1/elements)+(xbar/Sxx));



