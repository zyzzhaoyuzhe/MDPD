%% caluclate posterior distribution (with activeset)
% Output: post (k by n)
% Inputs:
% z: the raw data (k by n by m). 
% C: the probability matrix (confusion matrix) for all components. (r by c
% by m)
% W: a diagonal matrix which specifies the probabilities for all
% components. (k_component by k_component)
% activeset(optional): specifies which dimensions are used in the
% calculate. By default, activeset include all dimensions.

function post = stageEM_posterior(z,C,W,activeset)
% 
z_tmp = z(:,:,activeset);
C_tmp = C(:,:,activeset);

post = MDPD_posterior(z_tmp,C_tmp,W);
end