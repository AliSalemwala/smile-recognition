function [a, z] = feed_forward(w, a, b, z)

%     sample = a{1};
    %First layer calculations
%     for i=1: 44
%         sum = 0;
%         for j = 1:44
%             sum = sum + w{2}(i, j)*sample(j, 1);
%         end
%         z{2}(i, 1) = sum  + b{2}(i, 1);
%         a{2}(i, 1) = sigmoid(sum  + b{2}(i, 1));
%     end

    z{2} = w{2}*a{1} + b{2};
    a{2} = (sigmoid(z{2}));
    
    %Second layer calculations    
%     sample = a{2};
%     for i=1: 44
%         sum = 0;
%         for j = 1:44
%             sum = sum + w{3}(i, j)*sample(j, 1);
%         end
%         z{3}(i, 1) = sum  + b{3}(i, 1);
%         a{3}(i, 1) = sigmoid(sum  + b{3}(i, 1));
%     end
    z{3} = w{3}*a{2} + b{3};
    a{3} = sigmoid(z{3});
    %Output layer calculations
%     sample = a{3};
%     sum = 0;
%     for i= 1:44
%         sum = sum + w{4}(i, 1)*sample(i, 1);        
%     end
%     z{4} = sum  + b{4}(1, 1);
%     a{4} = sigmoid(sum  + b{4});
    z{4} = w{4}*a{3} + b{4};
    a{4} = sigmoid(z{4});
end

function v = sigmoid(x)
   v = 1./(1+exp(-x));
end

