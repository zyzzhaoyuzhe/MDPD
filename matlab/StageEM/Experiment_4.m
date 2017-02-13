%% Experment 4
% Dog dataset. 4 classes
clear
load('/media/vincent/Data/Dataset/dog/data.mat');

% use only 52 works
z = z(:,:,freq(1:52,1));


m = size(z,3);
n = size(z,2);
c = size(z,1);

%%
CS = CS_Spec(m,n,c);
CS.z = z;
CS.label = label;

%%
for i = 1:10
    CS.learn;
    [~,err_SPEC(i)] = CS.predict;
end

%%


%% Need to deal with missing data. (contains missing data)
% tip: stepsize = .048.
clear SEM
SEM = CS_stageEM_Ext(CS.m,CS.n,CS.c,CS.c+1);
SEM.MissingData(z);
SEM.label = CS.label;

SEM.Mstep_switch=2;

[SEM,foo] = learn(SEM);
SEM.LabelSwap([1 4 3 2]);
[pred_SEM,err_SEM] = SEM.predict;
%%
SEM.refine;
fullset = 1:SEM.m;
[pred_SEM_fine,err_SEM_fine] = SEM.predict(fullset);
4