%% Calculate Hessian matrix used to split components (ext)
% (ext) is for missing data mode. The last alphabet is missing data
% percentage.

% OUTPUT:
% H: Hessian matrix (2*(r-1) by 2*(r-1))
% INPUT:
% z: raw data
% C_tmp: 
% W_tmp:
% activeset: current activeset;
% k: split component k
% i,j: at i j coordinates


function H = stageEM_Hessian_ext(z,C_tmp,W_tmp,activeset,i,j,k)

r = size(z,1);
c = size(C_tmp,2);
%% split
C_tmp(:,end+1,:) = C_tmp(:,k,:);
W_tmp(end+1,end+1) = W_tmp(k,k)/2;
W_tmp(k,k) = W_tmp(k,k)/2;

% add into activeset
activeset(end+1) = i;
activeset(end+1) = j;
activeset = unique(activeset,'stable');
% old conditional mutual information;
oldMI = stageEM_MI(z,C_tmp,W_tmp,activeset);
oldMI = oldMI(i,j);

eps = 0.001;
H = zeros(2*(r-2),2*(r-2));
%

for a = 1:2*(r-2)
    if a > r-2
        li = a+2;
    else
        li = a;
    end
    parfor b = 1:2*(r-2)
        if b > r-2
            lj = b+2;
        else
            lj = b;
        end
        
        dmu = zeros(2*r,1);
        dmu(li) = eps;
        dmu(lj) = eps;
        if li == lj
            len = eps;
        else
            len = sqrt(2)*eps;
        end
        
        dmu(r-1) = -sum(dmu(1:r-2));
        dmu(2*r-1) = -sum(dmu(r+1:2*r-2));
             
        C_new1 = C_tmp;
        C_new2 = C_tmp;
        
        C_new1(:,k,i) = C_new1(:,k,i) + dmu(1:r);
        C_new1(:,end,i) = C_new1(:,end,i) - dmu(1:r);
        C_new1(:,k,j) = C_new1(:,k,j) + dmu(r+1:2*r);
        C_new1(:,end,j) = C_new1(:,end,j) - dmu(r+1:2*r);
        
        
        C_new2(:,k,i) = C_new2(:,k,i) - dmu(1:r);
        C_new2(:,end,i) = C_new2(:,end,i) + dmu(1:r);
        C_new2(:,k,j) = C_new2(:,k,j) - dmu(r+1:2*r);
        C_new2(:,end,j) = C_new2(:,end,j) + dmu(r+1:2*r);
        

        
        newMI1 = stageEM_MI(z,C_new1,W_tmp,activeset);
        newMI1 = newMI1(i,j);
        newMI2 = stageEM_MI(z,C_new2,W_tmp,activeset);
        newMI2 = newMI2(i,j);
        
        H(a,b) = (newMI1+newMI2-2*oldMI)/(len^2);
    end
end
end