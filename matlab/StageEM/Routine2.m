%% Routine 2
% Binary, random confusion matrix with diagonal dominance.

clear

m = 100;
n = 1000;
c = 3;

% CS = CS_Spec(m,n,c);

[W,C] = CWgen_multi_rand(m,c,[.6 .7]);


% CS.Wgen = W;
% CS.Cgen = C;
% CS.DataGen;

% benchmark
trumodel = MDPD_stageEM(m,n,c,c);
trumodel.Get_Para(C,W);
trumodel.Cgen = C;
trumodel.Wgen = W;
trumodel.DataGen;

[~,err_ben] = trumodel.predict(1:trumodel.m);


%% spectral method
CS = CS_Spec(m,n,c);
CS.z = trumodel.z;
CS.label = trumodel.label;
CS.learn;
[poo,err_SPEC] = CS.predict;
% if err_SPEC>0.5
%     err_SPEC = 1-err_SPEC;
% end
%%
model = MDPD_stageEM(m,n,c,c);
model.z = CS.z;
model.Cgen = CS.Cgen;
model.Wgen = CS.Wgen;
model.label = CS.label;
%%
[model,disp] = learn(model,'display','off');
model = MDPD_align(trumodel,model);
[model_predict,err_SEM] = model.predict;
1
length(model.activeset)

%% fine tune
model.refine;
fullset = 1:model.m;
[~,err_SEM_fine] = model.predict(fullset);

