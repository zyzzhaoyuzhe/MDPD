classdef CrowdSourcing < handle
    %%
    
    %% Input Data
    properties
        %% Index i for workers, j for items (questions), l for labels
        m = 0 % # of workers
        n = 0 % # of items
        c = 0 % # of labels (components)
        z %  k by n by m array. Use binary encoding for workers' answers (raw data)
        C % k by k by m. Confusion matrix for each worker
        W % diagonal matrix for label probablity
    end
    %% Generative Model (for generating data)
    properties
        Wgen
        Cgen % k by k by m confusion matrices
        label % true label
    end
    
    %%
    methods
        %% constructor
        function obj = CrowdSourcing(m,n,k)
            obj.m = m;
            obj.n = n;
            obj.c = k;
        end
    end
    
    %% Generating data
    methods
        %% Data Generation (ref: Yuchen's Paper on CrowdSourcing)
        function obj = DataGen(obj)
            if isempty(obj.Cgen) || isempty(obj.Wgen)
                display('no data generated');
                return
            end
            % Generate Data
            obj.Get_Label;
            pi = 1; % probablity of a worker labeling for an item
            obj.z = zeros(obj.c,obj.n,obj.m);
            for i = 1:obj.m
                for j = 1:obj.n
                    if rand < pi % label the item
                        prob = obj.Cgen(:,obj.label(j),i);
                        randnum = rand; % a rand number
                        
                        cumprob = cumsum(prob);
                        
                        idx = find(cumprob>randnum);
                        obj.z(idx(1),j,i) = 1;
                    end
                end
            end
            
        end
        %% Generate true label
        function obj = Get_Label(obj)
            randnum = rand(obj.n,1);
            cumprob = cumsum(diag(obj.Wgen));
            obj.label = ones(obj.n,1);
            for l = 1:obj.c-1
                idx = randnum>cumprob(l);
                obj.label(idx) = obj.label(idx)+1;
            end
        end
        %% missing data
        function obj = Get_MissingData(obj,miss)
            % miss is the percentage of missing data (default is 0);
            foo = rand(obj.n,obj.m);
            foo = foo<miss;
            for i = 1:obj.m
                bar = foo(:,i)';
                obj.z(:,bar==1,i) = 0;
            end
        end
    end
    
    %% Predicting and Performance
    methods
        function [output,err] = predict(obj)
            post = obj.posterior;
            [output,err] = MDPD_predict(post,obj.label);
        end
        %% posterior distributions of labels for all items given worker's answer
        function post = posterior(obj)
            post = MDPD_posterior(obj.z,obj.C,obj.W);
        end
    end
end