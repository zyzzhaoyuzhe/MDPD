%% Routine 3
% Binary, random confusion matrix, sparse (highly noisy case).

clear

CS = CS_Spec(100,1000,2);
%
% [W,C] =  CWgen_bin_sparse(CS.m,.5);
[W,C] =  CWgen_bin_sparse(CS.m,.10,'diagonal');

CS.Wgen = W;
CS.Cgen = C;
CS.DataGen;

% benchmark
CS.W = CS.Wgen;
CS.C = CS.Cgen;
[~,err_ben] = CS.predict;

%% spectral method
CS.learn;
[~,err_SPEC] = CS.predict;

%%
clear SEM;
SEM = CS_stageEM(CS.m,CS.n,CS.c);
SEM.z = CS.z;
SEM.Cgen = CS.Cgen;
SEM.Wgen = CS.Wgen;
SEM.label = CS.label;
%%
[SEM,foo] = learn(SEM);
[~,err_SEM] = SEM.predict;
1
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