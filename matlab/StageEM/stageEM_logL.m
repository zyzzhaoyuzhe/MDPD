%% Calculate log likelihood
% z: raw data(k by n by m)
% W: diagonal matrix (k by k)
% C: confusion matrix (k by k by m)

function output = stageEM_logL(z,W,C,activeset)
%
narginchk(4,4);
%
z = z(:,:,activeset);
C = C(:,:,activeset);

% 
c = size(W,1);
m = size(z,3);
n = size(z,2);
% foo (n by k by m)
foo = zeros(n,c,m);
for i = 1:m
    foo(:,:,i) = z(:,:,i)'*C(:,:,i);
end
foo = log(foo);
foo = sum(foo,3);

W = diag(W);
W = log(W(:));

foo = bsxfun(@plus,foo,W');
foo = logsum(foo,2);

output = sum(foo)/n;
end