%% Experment 1
% Binary Crowdsourcing setting. Random Confusion matrix with deceasing
% diagonal dominance. Compare spectral method with our algorithm.

%%
clear
try
    foo = feature('numCores');
    matlabpool(foo);
catch
end
%%
m = 100;
n = 1000;
r = 2;
c = 2;

%% init
nbin = 25;
nrepeat = 10;
problow = linspace(.4,.25,nbin);
probhigh = linspace(.9,.75,nbin);
range = [problow', probhigh'];

%%
Spec_output = zeros(nbin, nrepeat);
SEM_output = zeros(nbin,nrepeat);
SEM_MI = zeros(nbin,nrepeat);
nactiveset = zeros(nbin,nrepeat);
SEM_fine_output = zeros(nbin,nrepeat);
SEM_fine_MI = zeros(nbin,nrepeat);
benchmark_output = zeros(nbin,nrepeat);
benchmark_MI = zeros(nbin,nrepeat);

for i = 1:nbin
    tic
    range(i,:)
    for j = 1:nrepeat
        CS = CS_Spec(m,n,c);
%         [W,C] = CWgen_bin_rand(m,range(i,:));
        [W,C] = CWgen_multi_rand(m,c,range(i,:));
        CS.Wgen = W;
        CS.Cgen = C;
        CS.DataGen;
        
        %% spectral method
        try
            tmp = [];
            for k = 1:10
                CS.learn;
                [~,error] = CS.predict;
                if error>0.5
                    error = 1-error;
                end
                tmp(k) = error;
            end
            Spec_output(i,j) = mean(tmp);
        catch
            Spec_output(i,j) = nan;
            display('run spec failed');
        end
        %% Stagewise EM and fine tune stagewise EM
        %%
        try
            SEM = MDPD_stageEM(CS.m,CS.n,CS.c,CS.c);
            SEM.read(CS.z,CS.label);
            SEM.Cgen = CS.Cgen;
            SEM.Wgen = CS.Wgen;
            %%
            [SEM,foo] = learn(SEM,'display','off');
            
            trumodel = MDPD(m,n,c,c);
            trumodel.Get_Para(CS.Cgen,CS.Wgen);
            SEM = MDPD_align(trumodel,SEM);
            
            
            [~,error] = SEM.predict;
            if error > .5
                error = 1-error;
            end
            SEM_output(i,j) = error;
            SEM_MI(i,j) = foo{2}(end);
            nactiveset(i,j) = length(SEM.activeset);
            
            %% fine tune
            fullset = 1:SEM.m;
            SEM.refine;
            [~,error] = SEM.predict(fullset);
            if error > .5
                error = 1-error;
            end
            SEM_fine_output(i,j) = error;
            tmp = stageEM_MI(SEM.z,SEM.C,SEM.W,fullset);
            SEM_fine_MI(i,j) = max(tmp(:));
        catch
            SEM_output(i,j) = nan;
            SEM_fine_output(i,j) = nan;
            display('run stage-EM failed');
        end
        %% Benchmark
        CS.W = CS.Wgen;
        CS.C = CS.Cgen;
        [~,error] = CS.predict;
        benchmark_output(i,j) = error;
        tmp = stageEM_MI(CS.z,CS.Cgen,CS.Wgen,1:CS.m);
        benchmark_MI(i,j) = max(tmp(:));
    end
    toc
end

% save('Experiment1_outputs');

1