%% Routine 4
% Multi, random confusion matrix with diagonal dominance.

clear
m = 100;
n = 1000;
c = 4;



CS = CS_Spec(m,n,c);

[W,C] = CWgen_multi_rand(m,c,[.3 .5]);


CS.Wgen = W;
CS.Cgen = C;
CS.DataGen;

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
%%
model = MDPD_stageEM(CS.m,CS.n,CS.c,CS.c);
model.z = CS.z;
model.Cgen = CS.Cgen;
model.Wgen = CS.Wgen;
model.label = CS.label;
%%
trumodel = MDPD(m,n,c,c);
trumodel.C = CS.Cgen;
turmodel.W = CS.Wgen;

[model,disp] = learn(model,'display','off');
model = MDPD_align(trumodel,model);
[result,err_SEM] = model.predict;
1
length(model.activeset)
%% fine tune
model.refine;
fullset = 1:model.m;
[~,err_SEM_fine] = model.predict(fullset);
if err_SEM_fine>.5
    err_SEM_fine = 1-err_SEM_fine;
end

