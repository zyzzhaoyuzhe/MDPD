%% Calculate conditional mutual information of all pairs 
% OUTPUT: MI is m by m
% INPUTS:
% z: raw data (r by n by m);
% C_tmp: (r by c by m);
% W_tmp: (c by c)
% activeset: specify the support


function MI = stageEM_MI(z,C,W,activeset)
r = size(z,1);
n = size(z,2);
m = size(z,3);
c = size(C,2);
post = stageEM_posterior(z,C,W,activeset);
[C_fs,W_fs] = stageEM_Mstep(z,post);
ratio = stageEM_ratio(z,post,W_fs,C_fs);

MIhelper = zeros(m,m,n,c);
for k = 1:c
    MIhelper(:,:,:,k) = get_MIhelper(z,ratio(:,:,:,:,k));
end

MIres = stageEM_MIres(MIhelper,W_fs,post);
for l = 1:c
    for a = 1:m
        MIres(a,a,l) = 0;
    end
end

for k = 1:c
    MIres(:,:,k) = MIres(:,:,k)*W_fs(k,k);
end
MI = sum(MIres,3);
MI = MI-diag(diag(MI));
end