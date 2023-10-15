function [X] = MIMOcapacity(H,P)
% MIMOCAPACITY returns the optimal signaling for MIMO channel with a total
% power constraint
%   H: the MIMO channel
%   P: the total power

[U, eigchan] = eig(H'*H,'vector');
[eigchan, ind] = sort(eigchan,'descend'); % sort eigenvalues in decreasing order for water filling algorithm
U = U(:, ind);

zeroeigchan = (eigchan<1e-6); % ignore small eigenvalues
eigchan(zeroeigchan)=[];
U(:,zeroeigchan) = [];

waterlevel = 1;
nEigchans = length(eigchan);

igamma = waterlevel./(eigchan);
temp = 0;
% water filling algorithm
for k = nEigchans:-1:1
    temp = (P+sum(1./eigchan(1:k))*waterlevel)/k;
    if ((temp-igamma(k))>0)
        break;
    end
end

power = max(temp-igamma,0)';
X = U*diag(power)*U';

end

