function output = mylog(input)
    output = log(input);
    output(~isreal(output)) = -35;
    output(isinf(output)) = -35;
%     output(isnan(output)) = -35;
end