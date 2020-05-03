function [error, est, beta] = LinearRegression(trainf, traint, testf, testt)
    X = [ones(size(trainf,1), 1) trainf];
    test = [ones(size(testf,1), 1) testf];
    
    % Linear Regression Equation
    beta = inv(transpose(X)*X)*transpose(X)*traint;
   
    % Calculating error on test data
    t_size = size(testf, 1);
    est = zeros(t_size, 1);    
    error = zeros(t_size, 1);
    for i=1:t_size
        est(i,1) = test(i, :)*beta;
        error(i,1) = abs(testt(i,1) - est(i,1));
    end
    
    %Plotting Histogram
    histogram(error);
    xlim([0 1]);
    xlabel('Error');
    ylabel('Instances');
    title('Error Histogram for Linear Regression');
end

