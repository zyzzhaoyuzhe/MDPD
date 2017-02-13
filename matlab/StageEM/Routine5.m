%% Routine 5
% Multi, random confusion matrix with diagonal dominance with missing data

clear
m = 100;
n = 1000;
c = 3;

CS = CS_Spec(m,n,c);

[W,C] = CWgen_multi_rand(m,c,[.3 .6]);


CS.Wgen = W;
CS.Cgen = C;
CS.DataGen;

% missing data
CS.Get_MissingData(.7);

% benchmark
CS.W = CS.Wgen;
CS.C = CS.Cgen;
[~,err_ben] = CS.predict;


%% spectral method
CS.learn;
[predic_SPEC,err_SPEC] = CS.predict;
if err_SPEC>0.5
    err_SPEC = 1-err_SPEC;
end
%% Has missing data
clear SEM;
SEM = CS_stageEM_Ext(CS.m,CS.n,CS.c,CS.c+1);
SEM.MissingData(CS.z);
SEM.Cgen = CS.Cgen;
SEM.Wgen = CS.Wgen;
SEM.label = CS.label;
%%
[SEM,foo] = learn(SEM);
[~,err_SEM] = SEM.predict;
1
length(SEM.activeset)
if err_SEM>.5
    err_SEM = 1-err_SEM;
end

%% fine tune
SEM.refine;
fullset = 1:SEM.m;
[~,err_SEM_fine] = SEM.predict(fullset);
if err_SEM_fine>.5
    err_SEM_fine = 1-err_SEM_fine;
end

