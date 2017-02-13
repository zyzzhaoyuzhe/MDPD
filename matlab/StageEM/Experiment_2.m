%% Experment 2
% Binary Crowdsourcing setting. Sparse situation, with fewer "experts".

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
nbin = 20;
nrepeat = 10;

sparsity = linspace(0.20,0.05,nbin);
%%
Spec_output = zeros(nbin, nrepeat);
SEM_output = zeros(nbin,nrepeat);
SEM_MI = zeros(nbin,nrepeat);
nactiveset = zeros(nbin,nrepeat);
SEM_fine_output = zeros(nbin,nrepeat);
SEM_fine_MI = zeros(nbin,nrepeat);
benchmark_output = zeros(nbin,nrepeat);
benchmark_MI = zeros(nbin,nrepeat);

parfor i = 1:nbin
    sparsity(i)
    
    
    for j = 1:nrepeat
        clear CS SEM;
        CS = CS_Spec(m,n,c);
        [W,C] = CWgen_bin_sparse(m,sparsity(i),'diagonal');
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
        %% Stagewise EM
        %%
        try
            SEM = MDPD_stageEM(CS.m,CS.n,CS.c,CS.c);
            SEM.z = CS.z;
            SEM.Cgen = CS.Cgen;
            SEM.Wgen = CS.Wgen;
            SEM.label = CS.label;
            %%
            [SEM,foo] = learn(SEM,'display','off');
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
        end
       %% Benchmark
        CS.W = CS.Wgen;
        CS.C = CS.Cgen;
        [~,error] = CS.predict;
        benchmark_output(i,j) = error;
        tmp = stageEM_MI(CS.z,CS.Cgen,CS.Wgen,1:CS.m);
        benchmark_MI(i,j) = max(tmp(:));
    end
end

save('Experiment2_outputs');


2
