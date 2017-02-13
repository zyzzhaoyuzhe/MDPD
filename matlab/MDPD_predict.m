%% MDPD: Calculate posterior probabilities
% output: output (1-n). err is error rate.

function [output,err] = MDPD_predict(post,label)

nsam = length(label);% sample size
[~,output] = max(post,[],1);
output = output(:);

%% need to fix the label align problem.
% c = length(unique(label));
% % foo = perms(1:c);
% foo = 1:c;
% % helper vector
% bar = zeros(size(label,1),c);
% for l = 1:c
%     bar(:,l) = idx==l;
% end
% N = size(foo,1);
% err_tmp = zeros(N,1);
% % helper
% TT = 1:c;
% TT = TT(:);
% for n = 1:N
%     tmp = bar(:,foo(n,:));
%     tmp = tmp*TT;
%     err_tmp(n) = sum(tmp-label~=0);
%     err_tmp(n) = err_tmp(n)/nsam;
% end
% % find the smallest one
% [~,index] = min(err_tmp);
% tmp = bar(:,foo(index,:));
% output = tmp*TT;

err = sum(output-label~=0);
err = err/nsam;

end