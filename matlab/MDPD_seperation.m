%% Evaluate the sepeartion of different components in MDPD (by KL divergence)
% output: A matrix containing pairwise Jensen-Shannon Divergence between
% components.
function output = MDPD_seperation(C)
c = size(C,2);
m = size(C,3);
r = size(C,1);

output = zeros(c);
for k1 = 1:c
    for k2 = 1:c
        prob1 = squeeze(C(:,k1,:));
        prob2 = squeeze(C(:,k2,:));
        
        output(k1,k2) = MDPD_JS(prob1,prob2);
    end
end
end