% Generate Binary Crowdsourcing model with diagonal dominance specified by
% [probhi,problow]


function [Wgen,Cgen] = CWgen_bin_rand(m,range)
%
narginchk(2,2);
% mixture weights
Wgen = eye(2)*0.5;
% confusion matrix
probhi = range(2);
problow = range(1);
temp = problow+(probhi-problow)*rand(2,m);
Cgen = zeros(2,2,m);
for i = 1:m
    Cgen(:,:,i) = diag(temp(:,i)) + flipud(diag(1-temp(:,i)));
end
end