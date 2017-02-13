function output = logsum(input,dim)
% Logsum performs the calculation of the logrithm of a summation of
% elements. output = log(sum over i (e^input(i)))
% dim specifies the dimension of the sum, default dim = 1. Calculate column
% sum

narginchk(1,2);
if nargin == 1
    dim = 1;
end

maxinput = max(input,[],dim);
input = bsxfun(@minus,input,maxinput);
input = exp(input);
output = bsxfun(@plus,log(sum(input,dim)),maxinput);