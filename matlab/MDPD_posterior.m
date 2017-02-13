%% MDPD: Calculate posterior probabilities
% output: post (k by n)
function post = MDPD_posterior(z,C,W)
% check dimensions
if size(C,1)~=size(z,1) || size(C,3)~=size(z,3)
    error('dimensions do not match: z, C');
end
if ~isequal(W,diag(diag(W)))
    error('W is not diagonal matrix');
end
if size(C,2)~=size(W,1)
    error('dimensions do not match: C, W');
end

% dimensions
n = size(z,2);
m = size(z,3);
k = size(C,2);

% q represents the postierior labels for items. k by n
qtemp = zeros(k,n,m);
for i = 1:m
    qtemp(:,:,i) = C(:,:,i)'*z(:,:,i);
end
qtemp = mylog(qtemp);
qtemp(qtemp<-35) = -35;
qtemp = sum(qtemp,3);

logW = log(diag(W));
logW = logW(:);
qtemp = bsxfun(@plus,qtemp,logW);
lsum = logsum(qtemp,1);
post = exp(bsxfun(@minus,qtemp,lsum));
end