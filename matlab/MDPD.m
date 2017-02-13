classdef MDPD < handle
    %%
    
    %% Input Data
    properties
        %% Index i for workers, j for items (questions), l for labels
        m = 0 % # of workers
        n = 0 % # of items
        c = 0 % # of labels (components)
        r = 0 % # of discrete alphabet (option: for StageEM class)
        z %  r by n by m array. Use binary encoding for workers' answers (raw data)
        C % r by c by m. Confusion matrix for each worker
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
        function obj = MDPD(m,n,c,r)
            obj.m = m;
            obj.n = n;
            obj.c = c;
            obj.r = r;
        end
        %% Assign Parameter
        function obj = Get_Para(obj,C,W)
            obj.C = C;
            obj.W = W;
            obj.m = size(C,3);
            obj.c = size(C,2);
            obj.r = size(C,1);
        end
        %% read data
        function obj = read(obj,z,label)
            narginchk(2,3);
            obj.m = size(z,3);
            obj.n = size(z,2);
            obj.r = size(z,1);
            obj.z = z;
            if nargin == 3
                obj.label = label;
            end
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
        %% Seperation
        function output = seperation(obj)
            output = MDPD_seperation(obj.C);
        end
    end
    %% Miscellaneous
    methods
        %% Swap Label
        function obj = LabelSwap(obj,order)
            if length(order) ~= obj.c
                error('length of the input vector is not equal to the number of components');
            end
            if length(unique(order))~=obj.c
                error('entries in the order are duplicated.');
            end
            obj.C = obj.C(:,order,:);
            foo = diag(obj.W);
            foo = foo(order);
            obj.W = diag(foo);
        end
        %% 
    end
end