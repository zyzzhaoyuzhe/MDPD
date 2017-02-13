%% Experment 5
% WEB dataset. 5 classes
clear
load('/media/vincent/Data/Dataset/web/data.mat');

% pick top bla workers in frequency
z = z(:,:,freq(1:150,1));

m = size(z,3);
n = size(z,2);
c = size(z,1);

%
CS = CS_Spec(m,n,c);
CS.z = z;
CS.label = label;

%%
CS.learn;
[pred_SPEC,err_SPEC] = CS.predict;

%%


%% Need to deal with missing data. (contains missing data)
% tip: stepsize = .3.
clear SEM
SEM = CS_stageEM_Ext(CS.m,CS.n,CS.c,CS.c+1);
SEM.MissingData(z);
SEM.label = CS.label;
[SEM,foo] = learn(SEM);
[pred_SEM,err_SEM] = SEM.predict;
%%
SEM.refine;
fullset = 1:SEM.m;
[pred_SEM_fine,err_SEM_fine] = SEM.predict(fullset);
if err_SEM_fine>.5
    err_SEM_fine = 1-err_SEM;
end
5