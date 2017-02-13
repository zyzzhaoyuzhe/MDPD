%% Experiment 6
% fully M step vs Partial M step

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
c = 4;

%% init
nbin = 8;
nrepeat = 10;
problow = linspace(.5,.2,nbin);
probhigh = linspace(.6,.3,nbin);
range = [problow', probhigh'];

%%
Spec_output = zeros(nbin, nrepeat);
SEM_full_output = zeros(nbin,nrepeat);
SEM_part_output = zeros(nbin,nrepeat);
nactiveset = zeros(nbin,nrepeat);
SEM_full_fine_output = zeros(nbin,nrepeat);
SEM_part_fine_output = zeros(nbin,nrepeat);
benchmark_output = zeros(nbin,nrepeat);
benchmark_MI = zeros(nbin,nrepeat);

% range = [.5,.6];

parfor i = 1:nbin
    tic
    range(i,:)
    for j = 1:nrepeat
        CS = CS_Spec(m,n,c);
        [W,C] = CWgen_multi_rand(m,c,range(i,:));
        CS.Wgen = W;
        CS.Cgen = C;
        CS.DataGen;
        
        % benchmark
        trumodel = MDPD(m,n,c,c);
        trumodel.Get_Para(CS.Cgen,CS.Wgen);
        trumodel.z = CS.z;
        trumodel.label = CS.label;
        
        [~,err_ben] = trumodel.predict;
        benchmark_output(i,j) = err_ben;
        
        
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
            display('failed to run spec');
        end
        %% full M step
        try
            model_fully = MDPD_stageEM(m,n,c,c);
            model_fully.Mstep_switch = 1;
            model_fully.z = CS.z;
            model_fully.Cgen = CS.Cgen;
            model_fully.Wgen = CS.Wgen;
            model_fully.label = CS.label;
            %%
            [model_fully,disp] = learn(model_fully,'display','off','stopcrit','number of iterations');
            model_fully = MDPD_align(trumodel,model_fully);
            [model_predict,err_SEM_full] = model_fully.predict;
            SEM_full_output(i,j) = err_SEM_full;
            
            %% fine tune
            model_fully.refine;
            fullset = 1:model_fully.m;
            [~,err_SEM_full_fine] = model_fully.predict(fullset);
            SEM_full_fine_output(i,j) = err_SEM_full_fine;
        catch
            SEM_full_fine_output(i,j) = nan;
            display('failed to run full stageEM');
        end
        %% partial M step
        try
            model_part = MDPD_stageEM(m,n,c,c);
            model_part.Mstep_switch = 2;
            model_part.z = CS.z;
            model_part.Cgen = CS.Cgen;
            model_part.Wgen = CS.Wgen;
            model_part.label = CS.label;
            %%
            [model_part,disp] = learn(model_part,'display','off','stopcrit','number of iterations');
            model_part = MDPD_align(trumodel,model_part);
            [model_predict,err_SEM_part] = model_part.predict;
            SEM_part_output(i,j) = err_SEM_part;
            
            
            %% fine tune
            model_part.refine;
            fullset = 1:model_part.m;
            [~,err_SEM_part_fine] = model_part.predict(fullset);
            SEM_part_fine_output(i,j) = err_SEM_part_fine;
        catch
            SEM_part_output(i,j) = nan;
            display('failed to run partly stageEM');
        end
    end
    toc
end
% save('Experiment6_outputs');
6
