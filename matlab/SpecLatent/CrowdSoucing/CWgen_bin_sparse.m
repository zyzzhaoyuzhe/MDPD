% Binary sparse
% OUTPUT: C_gen and W_gen
% INPUT: sparsity: the sparsity level in percentage
function [Wgen,Cgen] = CWgen_bin_sparse(m,sparsity,option)
narginchk(2,3);
% mixture weights
Wgen = eye(2)*0.5;
% confusion matrix
nexp = floor(m*sparsity); % number of workers who are able to tell the difference of the true label

randhigh = .8;
randlow = .2;
Cgen(1,1,:) = randlow+(randhigh-randlow)*rand(1,1,m);
Cgen(2,1,:) = 1-Cgen(1,1,:);
Cgen(:,2,:) = Cgen(:,1,:);

exphi = .9;
explow = .6;

exprand = randi(2,nexp,1);

if nargin == 3
    if strcmp(option,'diagonal');
        exprand = ones(size(exprand));
    end
end

for i = 1:nexp
    if exprand(i) == 1
        foo = explow+(exphi-explow)*rand(2,1);
        template = diag(foo) + flipud(diag(1-foo));
    elseif exprand(i) == 2
        foo = explow+(exphi-explow)*rand(2,1);
        template = diag(1-foo) + flipud(diag(foo));
    end
    Cgen(:,:,i) = template;
end
end