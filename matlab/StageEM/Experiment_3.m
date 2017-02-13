%% Experment 3
% Binary Crowdsourcing bluebird dataset.
clear
load('/media/vincent/Data/Dataset/Bluebird/cubam-public/demo/bluebirds/data.mat');

m = size(z,3);
n = size(z,2);
c = size(z,1);

%%
CS = CS_Spec(m,n,2);
CS.z = z;
CS.label = label;

%%
for i = 1:10
    CS.learn;
    [~,err_SPEC(i)] = CS.predict;
end

%%
SEM = MDPD_stageEM(CS.m,CS.n,CS.c,CS.c);
SEM.Mstep_switch = 2;
SEM.z = CS.z;
SEM.label = CS.label;
[SEM,foo] = learn(SEM);
[~,err_SEM] = SEM.predict;
%%
SEM.refine;
fullset = 1:SEM.m;
[~,err_SEM_fine] = SEM.predict(fullset);
if err_SEM_fine>.5
    err_SEM_fine = 1-err_SEM;
end
3