%% Stagewise EM for crowdsourcing

classdef CS_stageEM < CrowdSourcing
    properties
        %% Heritaged from CrowdSourcing
%         %% Index i for workers, j for items (questions), l for labels
%         m = 0 % # of workers
%         n = 0 % # of items
%         c = 0 % # of labels
%         z % (data) k by n by m array. Use binary encoding for workers' answers (raw data)
%         C % (output) k by k by m. Confusion matrix for each worker
%         W % (output) diagonal matrix for label probablity
        %%
        activeset % active set for the variable selection
        c_tmp % temporary # of components
        C_tmp % tmp file for output C
        W_tmp % tmp file for output W
        MIres % m by m by k matrix. Mutual information matrix conditional on components (should be positive)
    end
    %% Helper properties
    properties
        ratio % f_S(x_i_x_j|y=k)/(f_S(x_i|y=k)f_S(x_j|y=k) (m by m by k by k by c).
        MIhelper % mutual information helper matrix: log(ratio) (m by m by n by c)
    end
    %%
    methods
        %% constructor
        function obj = CS_stageEM(m,n,k)
            obj@CrowdSourcing(m,n,k);
        end
        %%
        function [obj,disp] = learn(obj)
            disp = [];
            % initilize EM from the single component model
            obj.c_tmp = 1;
            obj.C_tmp = mean(obj.z,2);
            obj.W_tmp = 1;
            % initialize active set
            obj.activeset = [];
            % E-M loop
            stop = false;
            
            % Measure: 1-component model
            oldL = stageEM_logL(obj.z,obj.W_tmp,obj.C_tmp,1:obj.m);
            post = stageEM_posterior(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
            [C_fs,W_fs] = stageEM_Mstep(obj.z,post);
            obj.ratio = stageEM_ratio(obj.z,post,W_fs,C_fs);
            obj.get_MIhelper; % re-arrange the ratio matrix.
 
            obj.MIres = stageEM_MIres(obj.MIhelper,W_fs,post);
            for l = 1:obj.c_tmp
                for a = 1:obj.m
                    obj.MIres(a,a,l) = 0;
                end
            end
            [maxvalue,k] = max(obj.MIres,[],3);
            [maxvalue,j] = max(maxvalue,[],2);
            [maxvalue,i] = max(maxvalue,[],1);
            minvalue = min(min(min(obj.MIres)));
            j = j(i);
            k = k(i,j);
            
            % Init: for stop criterion
            diff = 0.005;
            oldmaxvalue = 10000;
            
            % Init: count
            count = 0;
            
%             %Init: disp
%             disp = cell(5,1);
%             disp{1}(1) = oldL;
%             disp{2}(1) = maxvalue;
%             disp{3}(1) = length(obj.activeset);
%             disp{4}(1) = minvalue;
%             disp{5}(:,:,1) = stageEM_MI(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
            
            while ~stop
                %
                count = count+1;
                % calculate current model posterior by using active set
                post = stageEM_posterior(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
                % update f_S(x_i_x_j|y=k) and arrange into (m by m by n)
                % according to the data
                [C_fs,W_fs] = stageEM_Mstep(obj.z,post);
                obj.ratio = stageEM_ratio(obj.z,post,W_fs,C_fs);
                obj.get_MIhelper; % re-arrange the ratio matrix.
                
                % update MIres and make diagonal entries zero
                obj.MIres = stageEM_MIres(obj.MIhelper,W_fs,post);
                for l = 1:obj.c_tmp
                    for a = 1:obj.m
                        obj.MIres(a,a,l) = 0;
                    end
                end
                
                % the biggest entry in MIres
                [maxvalue,k] = max(obj.MIres,[],3);
                [maxvalue,j] = max(maxvalue,[],2);
                [maxvalue,i] = max(maxvalue,[],1);
                j = j(i);
                k = k(i,j);
                
                
                % update the active set and split components (if necessary)
                if ~ismember(i,obj.activeset) || ~ismember(j,obj.activeset)
                    % add non member into activeset
                    if ~ismember(i,obj.activeset), obj.activeset(end+1) = i;end
                    if ~ismember(j,obj.activeset), obj.activeset(end+1) = j;end
                    if obj.c_tmp < obj.c % needs to split a component
                        if obj.c == 2
                            obj.split2(i,j,k);
                        else
                            obj.split_multi(i,j,k);
                        end
                    end
                end
                % EM with activeset
                obj.EM(obj.activeset);
                
%                 % measure performance (display)
%                 newL = stageEM_logL(obj.z,obj.W_tmp,obj.C_tmp,1:obj.m);
%                 disp{1}(end+1) = newL;
%                 disp{2}(end+1) = maxvalue;
%                 disp{3}(end+1) = length(obj.activeset);
%                 minvalue = min(min(min(obj.MIres(obj.activeset,obj.activeset,:))));
%                 disp{4}(end+1) = minvalue;
%                 disp{5}(:,:,end+1) = stageEM_MI(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
                
                % check stop criterion
                newL = stageEM_logL(obj.z,obj.W_tmp,obj.C_tmp,1:obj.m);
                if newL-oldL<diff && count > 20
                    stop = true;
                end
                oldL = newL;
                oldmaxvalue = maxvalue;
                
%                 %%%%%%%%%%%%%%%%%%%%%% (for analysis)
%                 oldL
%                 obj.activeset
%                 maxvalue
            end
            obj.C = obj.C_tmp;
            obj.W = obj.W_tmp;
        end
        
        %% model refinement
        function obj = refine(obj)
            niter = 20;
            for i = 1:niter
                % EM with fullset
                fullset = 1:obj.m;
                obj.EM(fullset);
                newL = stageEM_logL(obj.z,obj.W_tmp,obj.C_tmp,1:obj.m);
                
                % for analsis
%                 newL
            end
        end
    end
    %% helper functions
    methods
        %% get MIhelper
        % m by m by k matrix. Mutual information matrix conditional on components
        function obj = get_MIhelper(obj)
            % implementated by mex
            c = size(obj.ratio,5);
            for k = 1:c
                obj.MIhelper(:,:,:,k) = get_MIhelper(obj.z,obj.ratio(:,:,:,:,k));
            end
            
            
%             %%%%%%%%%%%%%%% old matlab implementation
%             % initiate
%             obj.MIhelper = zeros(obj.m,obj.m,obj.n);
%             %%%%%%%%%%
%             h = waitbar(0,'MIhelper: waiting');
%             for j = 1:obj.n
%                 waitbar(j/obj.n,h);
%                 for a = 1:obj.m
%                     for b = 1:obj.m
%                         l1 = obj.z(:,j,a);
%                         l1 = find(l1);
%                         l2 = obj.z(:,j,b);
%                         l2 = find(l2);
%                         tmp = obj.CO(a,b,l1,l2);
%                         obj.MIhelper(a,b,j) = log(tmp);
%                     end
%                 end
%             end
%             close(h);
        end
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Split component k into two components in (i,j) pair
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % two components
        function obj = split2(obj,i,j,k)
            narginchk(4,4);
            % split component k into two
            obj.C_tmp(:,end+1,:) = obj.C_tmp(:,k,:);
            obj.W_tmp(end+1,end+1) = obj.W_tmp(k,k)/2;
            obj.W_tmp(k,k) = obj.W_tmp(k,k)/2;
            obj.c_tmp = obj.c_tmp+1;
            % break the symmetry
            foo = 0;
            for c1 = 1:2
                for c2 = 1:2
                    if c1 == c2
                        foo = foo+obj.ratio(i,j,c1,c2,k);
                    else
                        foo = foo-obj.ratio(i,j,c1,c2,k);
                    end
                end
            end
            delta = 0.05;
            if foo > 0 % positively correlated
                obj.C_tmp(1,k,i) = obj.C_tmp(1,k,i) + delta;
                obj.C_tmp(2,k,i) = obj.C_tmp(2,k,i) - delta;
                obj.C_tmp(1,end,i) = obj.C_tmp(1,end,i) - delta;
                obj.C_tmp(2,end,i) = obj.C_tmp(2,end,i) + delta;
                
                obj.C_tmp(1,k,j) = obj.C_tmp(1,k,j) + delta;
                obj.C_tmp(2,k,j) = obj.C_tmp(2,k,j) - delta;
                obj.C_tmp(1,end,j) = obj.C_tmp(1,end,j) - delta;
                obj.C_tmp(2,end,j) = obj.C_tmp(2,end,j) + delta;
            else % negatively correlated
                obj.C_tmp(1,k,i) = obj.C_tmp(1,k,i) + delta;
                obj.C_tmp(2,k,i) = obj.C_tmp(2,k,i) - delta;
                obj.C_tmp(1,end,i) = obj.C_tmp(1,end,i) - delta;
                obj.C_tmp(2,end,i) = obj.C_tmp(2,end,i) + delta;
                
                obj.C_tmp(1,k,j) = obj.C_tmp(1,k,j) - delta;
                obj.C_tmp(2,k,j) = obj.C_tmp(2,k,j) + delta;
                obj.C_tmp(1,end,j) = obj.C_tmp(1,end,j) + delta;
                obj.C_tmp(2,end,j) = obj.C_tmp(2,end,j) - delta;
            end
        end
        
        % multi components
        function obj = split_multi(obj,i,j,k)
            % Calculate the Hessian of conditional mutual informaton and
            % find the most negative eigenvalue and its eigenvectors.
            H = stageEM_Hessian(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset,i,j,k);
            r = obj.c;
            % eigendecomposition. find most negative eigenvector
            [V,D] = eig(H);
            D = diag(D);
            [~,idx] = min(D);
            pertdir = V(:,idx); % 2*(r-1) dimensional
            % split component k into two
            obj.C_tmp(:,end+1,:) = obj.C_tmp(:,k,:);
            obj.W_tmp(end+1,end+1) = obj.W_tmp(k,k)/2;
            obj.W_tmp(k,k) = obj.W_tmp(k,k)/2;
            obj.c_tmp = obj.c_tmp+1;
            
            %%% for analysis
            oldMI = stageEM_MI(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
            oldMI = oldMI(i,j);
            %%%
            
            
            % break the dimension in the direction of pertdir
            % size of perturbation
            stepsize = 0.05;
            %
            dmui = pertdir(1:r-1);
            dmui(end+1) = -sum(dmui);
            dmui = dmui/norm(dmui)*stepsize;
            dmui = dmui-sum(dmui);
            
            
            dmuj = pertdir(1:r-1);
            dmuj(end+1) = -sum(dmuj);
            dmuj = dmuj/norm(dmuj)*stepsize;
            dmuj = dmuj-sum(dmuj);

            obj.C_tmp(:,k,i) = obj.C_tmp(:,k,i) + dmui;
            obj.C_tmp(:,end,i) = obj.C_tmp(:,end,i)-dmui;
            obj.C_tmp(:,k,j) = obj.C_tmp(:,k,j)+dmuj;
            obj.C_tmp(:,end,j) = obj.C_tmp(:,end,j)-dmuj;
            
            %%% for analysis
            newMI = stageEM_MI(obj.z,obj.C_tmp,obj.W_tmp,obj.activeset);
            newMI = newMI(i,j);
            %%%
        end
        
        %% EM
        function obj = EM(obj,activeset)
            % E step (calculate posteriors with the active set)
            post = stageEM_posterior(obj.z,obj.C_tmp,obj.W_tmp,activeset);
            
            % M step
            [obj.C_tmp,obj.W_tmp] = stageEM_Mstep(obj.z,post);
        end
        
        %% deal with missing data
        function obj = MissingData(obj,z)
            m = size(z,3);
            n = size(z,2);
            r = size(z,1);
            % add a new label
            newz = zeros(r+1,n,m);
            newz(1:end-1,:,:) = z;
            for i = 1:m
                foo = sum(z(:,:,i),1);
                idx = foo==0;
                newz(end,idx,i) = 1;
            end
            obj.z = newz;
        end
    end
    %% performance
    methods
        %% predict
        function [output,err] = predict(obj,SET)
            % SET specifies the coordinates to calculate the posterior
            % distribution.
            narginchk(1,2);
            if nargin == 1
                SET = obj.activeset;
            end
            
            post = stageEM_posterior(obj.z,obj.C,obj.W,SET);
            [output,err] = MDPD_predict(post,obj.label);
        end
        %% compare
    end
end