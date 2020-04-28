function [Return] = PriceToReturn(Price)
%The functions compute returns of a vector/matrix of price
% INPUT : Price = Vector of price

% OUTPUT : Return = Vector of returns
[Nrow,Ncol] = size(Price);

Return = zeros(Nrow-1,Ncol);

% Loop computing the returns
for i = 1:Ncol
    for j = 2:Nrow
        Return(j-1,i) = Price(j,i)./Price(j-1,i) -1;
    end
end

end

