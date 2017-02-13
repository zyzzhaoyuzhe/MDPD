%% Calculate mutual information of all pairs conditional on components
% OUTPUT: MIres is m by m by k
% INPUTS:
% MIhelper (m by m by n by c):
% log(f_S(x_i_x_j|y=k)/(f_S(x_i|y=k)f_S(x_j|y=k)) for all data.

% post (c by n): posterior distribution
% W: the prior probabilities for components. (c by c diagonal matrix).
% z: the original data (k by n by m)
% C: the confusion matrix (r by c by m)
% C_fs: the M-step of current model (r by c by m)

function MIres = stageEM_MIres(MIhelper,W_fs,post)
narginchk(3,3);
c = size(post,1);
n = size(MIhelper,3);
m = size(MIhelper,1);

% Init
MIres = zeros(m,m,c);

for k = 1:c
    weight = post(k,:)'/W_fs(k,k); 
%     weight = post(k,:)'; % modified 3-26-2016
    %
    temp = MIhelper(:,:,:,k);
    for j = 1:n
        temp(:,:,j) = temp(:,:,j)*weight(j);
    end
    MIres(:,:,k) = mean(temp,3);
end
end
