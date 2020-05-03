
clc;
clear;
%% Code

load('data.mat');
m = (totMatrix(randperm(size(totMatrix,1)),:));
[i, j] = size(m);
m(:, j) = m(:, j)/100;
x = normc(m(:, 1:j-1));
% 30371 test samples, 91113 training samples
train = x(1:i*0.75,:); %75 percent
test = x(i*0.75+1:i ,:); %25 percent

train_features = train(:,1:44);
train_targets = m(1:i*0.75, 45);
test_features = test(:,1:44);
test_targets = m(i*0.75+1:i, 45);

%% Neural Networks
[err, w, b, est, z, a] = NeuralNetworks(train_features, train_targets, test_features, test_targets);

%% Linear Regression
% [err, est, beta] = LinearRegression(train_features, train_targets, test_features, test_targets);

%% Scatter Plot
figure(2);
scatter((1:1:size(test_features, 1)), err);
xlabel('Nth observation');
ylim([0 1]);
ylabel('Error');
title('Neural Network Error Plot')

%% Accuracy Calculation
total = size(est,1);
predicted = 0;
for i=1:size(est,1)
    if est(i, 1) > 0.6
        if test_targets(i, 1) > 0.6
            predicted = predicted + 1;
        end        
    else 
        if test_targets(i, 1) <= 0.6
            predicted = predicted + 1;
        end
    end            
end
accuracy = predicted/total;
%[err,w, pred] = SimpleNN(train, test);

