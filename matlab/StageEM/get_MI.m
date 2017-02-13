% dmu shoud be (2*r by 1)

function output = get_MI(dmu,z,C_tmp,W_tmp,activeset,i,j,k)

newactiveset = [activeset,i,j];


r = size(C_tmp,1);
% split component k into two (duplicate)
C_tmp(:,end+1,:) = C_tmp(:,k,:);
W_tmp(end+1,end+1) = W_tmp(k,k)/2;


% perturbute by dmu
dmui = dmu(1:r);
dmuj = dmu(r+1:end);
C_tmp(:,k,i) = C_tmp(:,k,i)+dmui;
C_tmp(:,end,i) = C_tmp(:,end,i)-dmui;
C_tmp(:,k,j) = C_tmp(:,k,j)+dmuj;
C_tmp(:,end,j) = C_tmp(:,end,j)-dmuj;


post = stageEM_posterior(z,C_tmp,W_tmp,newactiveset);
% update f_S(x_i_x_j|y=k) and arrange into (m by m by n)
% according to the data
[C_fs,W_fs] = stageEM_Mstep(z,post);
ratio = stageEM_ratio(z,post,W_fs,C_fs);
c = size(C_tmp,2);
for k = 1:c
    MIhelper(:,:,:,c) = get_MIhelper(z,ratio);
end

% update MIres and make diagonal entries zero
MIres = stageEM_MIres(MIhelper,W_fs,post);

output = MIres(i,j,:);
output = squeeze(output);
output = dot(output,diag(W_fs));