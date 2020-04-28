%% Lilliefors test

function [max_test]=Lilliefors(x)
    [L,K]=size(x);
    max_test=zeros(1,K);
    for i=1:K
        s_x=sort(x(:,i));
        p=normcdf(s_x);
        test=abs(p-1/L);
        max_test(1,i)= max(test);
    end
end
    