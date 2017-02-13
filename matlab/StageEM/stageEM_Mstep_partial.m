%% M step (partial) for stageEM
% post is the posterior distribution obtained from the E step.


function [C,W] = stageEM_Mstep_partial(z,oldC,post,activeset)
%
m = size(z,3); % dimensions
c = size(post,1); % # components
r = size(z,1); % # labels

% update confusion matrix C
C = oldC;
for i = activeset
    temp = z(:,:,i)*post';
    temp = bsxfun(@rdivide,temp,sum(temp,1));
    C(:,:,i) = temp;
end
% update the distribution of labels
temp = sum(post,2);
W = diag(temp/sum(temp));
end