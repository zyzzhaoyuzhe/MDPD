%% Evaluate the sepeartion of different two product discrete distribution
% INPUT: prob1, prob2 are r by m
% OUTPUT: scalar (Jensen-Shannon Divergence)
function output = MDPD_JS(prob1,prob2)
prob1 = squeeze(prob1);
prob2 = squeeze(prob2);

aveprob = 1/2*(prob1+prob2);

foo1 = sum(sum(prob1.*log(prob1./aveprob),1));
foo2 = sum(sum(prob2.*log(prob2./aveprob),1));

output = 1/2*(foo1+foo2);
end