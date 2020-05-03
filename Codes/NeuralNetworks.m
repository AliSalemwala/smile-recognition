function [error, w, b, est, z, a] = NeuralNetworks( trainf, traint, testf, testt)
%Neural Network Implementation for 2 layers

    tic;
%% Initializations

    %for layer 1
    z{1} = ones(44, 1);
    w{2} = ones(44);
    b{2} = ones(44, 1);
    for i=1:44
        b{2}(i, 1) = rand;
        for j=1:44
            w{2}(i, j) = rand;
        end
    end
    
    %for layer 2
    z{2} = ones(44, 1);
    w{3} = ones(44);
    b{3} = ones(44, 1);
    for i=1:44
        b{3}(i, 1) = rand;
        for j=1:44
            w{3}(i, j) = rand;
        end
    end
    
    %for Output layer
    z{3} = 1;
    w{4} = ones(44, 1);
    b{4} = rand;
    for i=1:44
        w{4}(i, j) = rand;
    end
    
    a{1} = zeros(44, 1);
    a{2} = zeros(44, 1);
    a{3} = zeros(44, 1);
    a{4} = zeros(1, 1);
%   END OF INITIALIZATIONS
%% Training Process

lr = 0.001;
trainf_size = size(trainf);
max_iters = (trainf_size(1)) * 10;
m = max_iters;
%% Layer 1 activation
    for choice = 1:max_iters
        r = mod(choice, trainf_size(1))+1;
        if r == 0
            r = mod(choice, trainf_size(1)) + 1;
        end
        sample = transpose(trainf(r, :));
        sample_target = (traint(r, 1));
        a{1} = sample;
        z{1} = sample;
        x = z{1};
        %% Feed Forward
        [a, z] = feed_forward(w, a, b);

        z{1} = x;
        %% Compute error vector
        o_err{4} = (a{4} - sample_target).*derivative_sigmoid(z{4});

        %% Back propogate
        for i=3:-1:2
            if i == 3
                temp = transpose(w{i+1}(:, 1));
                o_err{i} = ((temp)*o_err{i+1}).*derivative_sigmoid(z{i});
            else
                o_err{i} = (transpose(w{i+1})*o_err{i+1}).*derivative_sigmoid(z{i});        
            end
        end

        %% Gradient Descent
        for i=4:-1:2
            if i == 4
                temp = lr*o_err{i}*transpose(a{i-1});
                w{4}(:, 1) = w{4}(:, 1) - temp(:, 1);
            else
                w{i} = w{i} - lr*o_err{i}*transpose(a{i-1});
            end
            b{i} = b{i} - lr*o_err{i};
        end
    end
    toc;
    b{4} = sum(b{4});
    t_size = size(testf, 1);
    est = zeros(t_size, 1);    
    error = zeros(t_size, 1);
    for i=1:t_size
        g = calcOutput(transpose(testf(i, :)), a, w, b);
        est(i, 1) = g;
        error(i,1) = abs(testt(i,1) - est(i,1));
    end
    histogram(error);
    xlabel('Error');
    ylabel('Instances');
    xlim([0 1]);
    title('Error Histogram for Neural Networks');
end

function v = sigmoid(x)
   v = 1./(1+exp(-x));
end

function v = derivative_sigmoid(x)  
  v = sigmoid(x).*(1-sigmoid(x));
end

function v = relu(x)
    if x < 0
        v = 0;
    else
        v = x;
    end
end
