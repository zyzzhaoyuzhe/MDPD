function [Mu,W] = TensorPower(M2,M3,k)
%% Perform Robust Tensor Power method for given 2nd and 3rd order symmetric tensor
n = size(M2,2);
dim = size(M2,1);
M2_matrix = 1/n*M2(:,:,1)*M2(:,:,2)'; % should be symmetic
[U,S,~] = svd(M2_matrix);

U = U(:,1:k);
S = S(1:k,1:k);
Q = U*diag(1./sqrt(diag(S))); % Q is the whitening matrix which should be dim by k

% white M3
M3_tilde = zeros(k,n,3);
for g = 1:3
    M3_tilde(:,:,g) = Q'*M3(:,:,g);
end

% initialization Mu and W
w = zeros(k,1); % w hat
Mu = zeros(dim,k); %  mu hat
lambda = zeros(k,1); % eigen value of tensor
v = zeros(k,1); % eigen vectors of tensor

% Tensor decomposition
tensor = M3_tilde;
L = 10;
N = 100;
for l = 1:k
    [s_temp,v_temp] = TensorPower_helper(tensor,L,N,lambda,v,l); % robust tensor decomposition
    lambda(l) = s_temp;
    v(:,l) = v_temp;
    
    w(l) = 1/s_temp^2;
    Mu(:,l) = s_temp*pinv(Q')*v_temp;
end
W = diag(w);

% re-order columns
[~,idx] = max(Mu,[],1);
[~,idx] = sort(idx);
if length(unique(idx))<k
    warning('unable to reorder columns of Mu');
    % let W MU intact.
elseif length(unique(idx))==k
    W = W(idx,idx);
    Mu = Mu(:,idx);
end

end