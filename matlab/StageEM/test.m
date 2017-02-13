tic
%%
m = 100;
n = 1000;
r = 2;
c = 2;
%%
truemodel = MDPD(m,n,c,r);
[W,C] = CWgen_multi_rand(m,c,[0.45,0.65]);
truemodel.Cgen = C;
truemodel.Wgen = W;
truemodel.DataGen;
truemodel.C = C;
truemodel.W = W;
[~, error] = truemodel.predict;
display(error)

SEM = MDPD_stageEM(m,n,c,c);
SEM.read(truemodel.z,truemodel.label);
[SEM,foo] = learn(SEM,'stopcrit','number of iterations');
SEM = MDPD_align(truemodel,SEM);
[~,error] = SEM.predict;
display(error)

SEM.refine;
[~,error] = SEM.predict(1:m);
display(error)

toc