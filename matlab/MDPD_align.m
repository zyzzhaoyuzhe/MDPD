%% Align the components of obj2 to the components of obj1
% OUTPUT: output aligned obj2

function output = MDPD_align(obj1,obj2)
if obj1.c~=obj2.c
    error('The number of components of two inputs are not equal.');
end
output = obj2;
c = obj1.c;
order = zeros(1,c);
for k1 = 1:c
    JS_min = 1000000;
    bar = nan;
    for k2 = 1:c
        if ~ismember(k2,order)
            foo = MDPD_JS(obj1.C(:,k1,:),obj2.C(:,k2,:));
            if foo<JS_min
                JS_min = foo;
                bar = k2;
            end
        end
    end
    order(k1) = bar;
end
output.LabelSwap(order);
end
