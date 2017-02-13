% Generate Binary Crowdsourcing model with diagonal dominance specified by
% [probhi,problow]
% INPUT: 
% m: # workers
% c: # labels
% range: level of diagonal dominance

function [Wgen,Cgen] = CWgen_multi_rand(m,c,range)
%
narginchk(3,3);
% mixture weights
Wgen = eye(c)/c;
% confusion matrix
probhi = range(2);
problow = range(1);
temp = problow+(probhi-problow)*rand(c,m);
Cgen = zeros(c,c,m);
for i = 1:m
    Cgen(:,:,i) = diag(temp(:,i));
    res = 1-sum(Cgen(:,:,i),1);
    %
%     ratio = rand(c-1,c);
%     ratio = bsxfun(@rdivide,ratio,sum(ratio,1));
%     ratio = bsxfun(@times,ratio,res);
    for k = 1:c
        Cgen([1:k-1 k+1:end],k,i) = ones(c-1,1)*res(1)/(c-1);
    end
    Cgen(:,:,i) = bsxfun(@rdivide,Cgen(:,:,i),sum(Cgen(:,:,i),1));
end
end