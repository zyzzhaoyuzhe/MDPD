%% Experment 1
% Binary Crowdsourcing setting. Random Confusion matrix with deceasing
% diagonal dominance. Compare spectral method with our algorithm.

%%
clear
try
    matlabpool 4;
catch
end
%%
m = 100;
n = 1000;
r = 5;
c = 5;

%% init
nbin = 25;
problow = linspace(.4,.25,nbin);
probhigh = linspace(.9,.75,nbin);
range = [problow', probhigh'];

%
nrepeat = 10;
Spec_output = zeros(nbin, nrepeat);
SEM_output = zeros(nbin,nrepeat);
nactiveset = zeros(nbin,nrepeat);
SEM_fine_output = zeros(nbin,nrepeat);
benchmark_output = zeros(nbin,nrepeat);

parfor i = 1:nbin
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
        end
        %% Stagewise EM and fine tune stagewise EM
        %%
        try
            SEM = MDPD_stageEM(CS.m,CS.n,CS.c,CS.c);
            SEM.z = CS.z;
            SEM.Cgen = CS.Cgen;
            SEM.Wgen = CS.Wgen;
            SEM.label = CS.label;
            %%
            [SEM,foo] = learn(SEM,'display','off');
            
            trumodel = MDPD(m,n,c,c);
            trumodel.Get_Para(CS.C,CS.W);
            SEM = MDPD_align(trumodel,SEM);
            
            
            [~,error] = SEM.predict;
            if error > .5
                error = 1-error;
            end
            SEM_output(i,j) = error;
            nactiveset(i,j) = length(SEM.activeset);
            
            %% fine tune
            fullset = 1:SEM.m;
            SEM.refine;
            [~,error] = SEM.predict(fullset);
            if error > .5
                error = 1-error;
            end
            SEM_fine_output(i,j) = error;
            
        catch
            
            SEM_output(i,j) = nan;
            SEM_fine_output(i,j) = nan;
        end
        %% Benchmark
        CS.W = CS.Wgen;
        CS.C = CS.Cgen;
        [~,error] = CS.predict;
        benchmark_output(i,j) = error;
    end
end

save('Experiment1_outputs','Spec_output','SEM_output','SEM_fine_output','benchmark_output');

1