classdef MDPD_stageEM < MDPD
    properties
        %% Heritaged from CrowdSourcing
        %         %% Index i for workers, j for items (questions), l for labels
        %         m = 0 % # of workers
        %         n = 0 % # of items
        %         c = 0 % # of labels
        %         r = 0 % # of discrete alphabet
        %         z % (data) r by n by m array. Use binary encoding for workers' answers (raw data)
        %         C % (output) r by c by m. Confusion matrix for each worker
        %         W % (output) diagonal matrix for label probablity
        
        activeset % active set for the variable selection
        c_tmp % temporary # of components
        MIres % m by m by k matrix. Mutual information matrix conditional on components (should be positive)
    end
    %% Helper properties
    properties
        ratio % f_S(x_i_x_j|y=k)/(f_S(x_i|y=k)f_S(x_j|y=k) (m by m by r by r by c).
        MIhelper % mutual information helper matrix: log(ratio) (m by m by n by c)
    end
    %% function switch (for easy function swap)
    properties
        %% M-step
        Mstep_switch = 1;
    end
    %%
    methods
        %% constructor
        function obj = MDPD_stageEM(m,n,c,r)
            obj@MDPD(m,n,c,r);
        end
        
        %%
        function [obj,disp] = learn(obj,varargin)
            % varargin
            p = inputParser;
            
            default_maxiter = 30;
            checkmaxiter = @(x) isnumeric(x);
            
            default_diffthre = 0.001;
            checkdiffthre = @(x) isnumeric(x);
            
            default_StopCrit = 'likelihood';
            valid_StopCrit = {'likelihood','mutualinformation','number of components','number of iterations'};
            checkStpCrit = @(x) any(validatestring(x,valid_StopCrit));
            
            default_num_c = 6;
            checknumc = @(x) isnumeric(x);
            
            default_numIter = 30;
            checknumIter = @(x) isnumeric(x);
            
            default_init = 'on';
            valid_init = {'on','off'};
            checkinit = @(x) any(validatestring(x,valid_init));
            
            default_display = 'on';
            valid_display = {'on','off'};
            checkdisplay = @(x) any(validatestring(x,valid_display));
           
            p.addParamValue('maxiter',default_maxiter,checkmaxiter);
            p.addParamValue('diffthre',default_diffthre,checkdiffthre);
            p.addParamValue('stopcrit',default_StopCrit,checkStpCrit);
            p.addParamValue('NumberComponents',default_num_c,checknumc);
            p.addParamValue('NumberIter',default_numIter,checknumIter);
            p.addParamValue('initialize',default_init,checkinit);
            p.addParamValue('display',default_display,checkdisplay);
            
            p.parse(varargin{:});
            
            % Creat stop crit function handle
            if strcmp(p.Results.stopcrit,'likelihood') % likelihood converges
                stop_func = @(input) input(1)-input(2)<p.Results.diffthre || input(3) > p.Results.maxiter; % input(1:3) = newL, oldL, count
            elseif strcmp(p.Results.stopcrit,'mutualinformation') % mutual information converges
                stop_func = @(input) input(1)-input(2)<p.Results.diffthre || input(3) > p.Results.maxiter; % 
            elseif strcmp(p.Results.stopcrit,'number of components')
                stop_func = @(input) input>=p.Results.NumberComponents;
            elseif strcmp(p.Results.stopcrit,'number of iterations')
                stop_func = @(input) input>=p.Results.NumberIter;
            end
                
            
            %
            delta = 0.002;
            % INIT
            if strcmp(p.Results.initialize,'on')
                obj.c_tmp = 1;
                obj.C = mean(obj.z,2);
                obj.W = 1;
                obj.C(obj.C<delta) = delta;
                obj.activeset = [];
            end
            
            % E-M loop
            STOP = false;
            
            % Init: for stop criterion
            oldmax = 10000;
            oldL = -Inf;
            
            % Init: count
            count = 0;
            
            %Init: disp
            flagdisplay = strcmp(p.Results.display,'on');
            
            disp = cell(5,1);
            disp{1}(1) = stageEM_logL(obj.z,obj.W,obj.C,1:obj.m); % log-likelihood
            
            post = stageEM_posterior(obj.z,obj.C,obj.W,obj.activeset);
            [C_fs,W_fs] = stageEM_Mstep(obj.z,post);
            obj.ratio = stageEM_ratio(obj.z,post,W_fs,C_fs);
            obj.get_MIhelper; % re-arrange the ratio matrix.
            obj.MIres = stageEM_MIres(obj.MIhelper,W_fs,post);
            for l = 1:obj.c_tmp
                for a = 1:obj.m
                    obj.MIres(a,a,l) = 0;
                end
            end
            for k = 1:obj.c_tmp
                obj.MIres(:,:,k) = obj.MIres(:,:,k)*W_fs(k,k);
            end
            MI = sum(obj.MIres,3);
            MI = MI-diag(diag(MI));
            disp{2}(1) = max(max(MI)); % max norm of MI
            disp{3}(1) = length(obj.activeset); % size of the active set
            disp{4}(1) = min(min(min(obj.MIres)));
            disp{5}(:,:,1) = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);

            %%%%%%% Body %%%%%%
            while ~STOP
                %
                count = count+1;
                %%% calculate current model posterior by using active set
                post = stageEM_posterior(obj.z,obj.C,obj.W,obj.activeset);
                %%% update f_S(x_i_x_j|y=k) and arrange into (m by m by n)
                % according to the data
                [C_fs,W_fs] = stageEM_Mstep(obj.z,post);
                obj.ratio = stageEM_ratio(obj.z,post,W_fs,C_fs);
                obj.get_MIhelper; % re-arrange the ratio matrix.
                
                %%% update MIres and make diagonal entries zero
                obj.MIres = stageEM_MIres(obj.MIhelper,W_fs,post);
                for l = 1:obj.c_tmp
                    for a = 1:obj.m
                        obj.MIres(a,a,l) = 0;
                    end
                end
                
                % the biggest entry in MIres
                [maxvalue,k] = max(obj.MIres,[],3);
                [maxvalue,j] = max(maxvalue,[],2);
                [~,i] = max(maxvalue,[],1);
                j = j(i);
                k = k(i,j);
                for k = 1:obj.c_tmp
                    obj.MIres(:,:,k) = obj.MIres(:,:,k)*W_fs(k,k);
                end
                
                %%% update the active set and split components (if necessary)
                if ~ismember(i,obj.activeset) || ~ismember(j,obj.activeset)
                    % add non member into activeset
                    if ~ismember(i,obj.activeset), obj.activeset(end+1) = i;end
                    if ~ismember(j,obj.activeset), obj.activeset(end+1) = j;end
                    if obj.c_tmp < obj.c % needs to split a component
                        obj.split_multi(i,j,k);
%                         obj.split_multi_rand(i,j,k);
                    end
                end
                % to prevent error in Calculating Hessian.(hard to perturb when
                % probability equals zero)
                obj.C(obj.C<delta) = delta;
                
                %%% EM with activeset
                obj.EM(obj.activeset);
                
                %%% measure performance
                newL = stageEM_logL(obj.z,obj.W,obj.C,1:obj.m);

                MI = sum(obj.MIres,3);
                MI = MI-diag(diag(MI));
                newmax = max(max(MI));
                
                disp{1}(end+1) = newL;
                disp{2}(end+1) = newmax;
                disp{3}(end+1) = length(obj.activeset);
                minvalue = min(min(min(obj.MIres(obj.activeset,obj.activeset,:))));
                disp{4}(end+1) = minvalue;
                disp{5}(:,:,end+1) = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);

                
                %%% check stop criterion
                if strcmp(p.Results.stopcrit,'likelihood') % likelihood converges
                    input = [newL,oldL,count];
                elseif strcmp(p.Results.stopcrit,'mutualinformation') % mutual information converges
                    input = [newmax,oldmax,count];
                elseif strcmp(p.Results.stopcrit,'number of components')
                    input = obj.c_tmp;
                elseif strcmp(p.Results.stopcrit,'number of iterations')
                    input = count;
                end
                
                if stop_func(input)
                    STOP = true;
                end
                
                oldL = newL;
                oldmax = newmax;
                
                %%% display on the screen
                if flagdisplay
                    count
                    oldL
                    oldmax
                    obj.activeset
                end
                
            end
        end
        
        %% model refinement
        function obj = refine(obj,niter)
            narginchk(1,2);
            if nargin == 1
                niter = 20;
            end
            for i = 1:niter
                % EM with fullset
                fullset = 1:obj.m;
                obj.EM(fullset);
                newL = stageEM_logL(obj.z,obj.W,obj.C,1:obj.m);
                
                % for debugging
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
            for k = 1:c %% can be parfor
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
        
        % binary responses
        function obj = split2(obj,i,j,k)
            narginchk(4,4);
            % split component k into two
            obj.C(:,end+1,:) = obj.C(:,k,:);
            obj.W(end+1,end+1) = obj.W(k,k)/2;
            obj.W(k,k) = obj.W(k,k)/2;
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
                obj.C(1,k,i) = obj.C(1,k,i) + delta;
                obj.C(2,k,i) = obj.C(2,k,i) - delta;
                obj.C(1,end,i) = obj.C(1,end,i) - delta;
                obj.C(2,end,i) = obj.C(2,end,i) + delta;
                
                obj.C(1,k,j) = obj.C(1,k,j) + delta;
                obj.C(2,k,j) = obj.C(2,k,j) - delta;
                obj.C(1,end,j) = obj.C(1,end,j) - delta;
                obj.C(2,end,j) = obj.C(2,end,j) + delta;
            else % negatively correlated
                obj.C(1,k,i) = obj.C(1,k,i) + delta;
                obj.C(2,k,i) = obj.C(2,k,i) - delta;
                obj.C(1,end,i) = obj.C(1,end,i) - delta;
                obj.C(2,end,i) = obj.C(2,end,i) + delta;
                
                obj.C(1,k,j) = obj.C(1,k,j) - delta;
                obj.C(2,k,j) = obj.C(2,k,j) + delta;
                obj.C(1,end,j) = obj.C(1,end,j) + delta;
                obj.C(2,end,j) = obj.C(2,end,j) - delta;
            end
        end
        
        % multi responses
        function obj = split_multi(obj,i,j,k)
            % Calculate the Hessian of conditional mutual informaton and
            % find the most negative eigenvalue and its eigenvectors.
            H = stageEM_Hessian(obj.z,obj.C,obj.W,obj.activeset,i,j,k);
            r = obj.r;
            % eigendecomposition. find most negative eigenvector
            [V,D] = eig(H);
            D = diag(D);
            [~,idx] = min(D);
            pertdir = V(:,idx); % 2*(r-1) dimensional (missing data mode 2*(r-2))
            
            % split component k into two
            obj.C(:,end+1,:) = obj.C(:,k,:);
            obj.W(end+1,end+1) = obj.W(k,k)/2;
            obj.W(k,k) = obj.W(k,k)/2;
            obj.c_tmp = obj.c_tmp+1;
            
            %%% for analysis
            oldMI = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);
            oldMI = oldMI(i,j);
            %%%
            
            
            % break the dimension in the direction of pertdir
            % size of perturbation
            stepsize = 0.05;
            dmui = pertdir(1:r-1);
            dmui(end+1) = -sum(dmui);
            dmui = dmui(:);
            
            dmuj = pertdir(r:end);
            dmuj(end+1) = -sum(dmuj);
            dmuj = dmuj(:);
            
            
            dmui = dmui/norm(dmui)*stepsize;
            dmuj = dmuj/norm(dmuj)*stepsize;

            obj.C(:,k,i) = obj.C(:,k,i) + dmui;
            obj.C(:,end,i) = obj.C(:,end,i)-dmui;
            obj.C(:,k,j) = obj.C(:,k,j)+dmuj;
            obj.C(:,end,j) = obj.C(:,end,j)-dmuj;            
            
            %%% for analysis
            newMI = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);
            newMI = newMI(i,j);
            %%%  
        end
        
        % multi responses (random split)
        function obj = split_multi_rand(obj,i,j,k)
            % Calculate the Hessian of conditional mutual informaton and
            % find the most negative eigenvalue and its eigenvectors.
            r = obj.r;
            % random perturbation direction
            pertdir = rand(2*(r-1),1); % 2*(r-1) dimensional (missing data mode 2*(r-2))
            
            % split component k into two
            obj.C(:,end+1,:) = obj.C(:,k,:);
            obj.W(end+1,end+1) = obj.W(k,k)/2;
            obj.W(k,k) = obj.W(k,k)/2;
            obj.c_tmp = obj.c_tmp+1;
            
            %%% for debugging
            oldMI = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);
            oldMI = oldMI(i,j);
            %%%
            
            
            % break the dimension in the direction of pertdir
            % size of perturbation
            stepsize = 0.05;
            dmui = pertdir(1:r-1);
            dmui(end+1) = -sum(dmui);
            dmui = dmui(:);
            
            dmuj = pertdir(1:r-1);
            dmuj(end+1) = -sum(dmuj);
            dmuj = dmuj(:);
            
            
            dmui = dmui/norm(dmui)*stepsize;
            dmuj = dmuj/norm(dmuj)*stepsize;

            obj.C(:,k,i) = obj.C(:,k,i) + dmui;
            obj.C(:,end,i) = obj.C(:,end,i)-dmui;
            obj.C(:,k,j) = obj.C(:,k,j)+dmuj;
            obj.C(:,end,j) = obj.C(:,end,j)-dmuj;            
            
            %%% for debugging
            newMI = stageEM_MI(obj.z,obj.C,obj.W,obj.activeset);
            newMI = newMI(i,j);
            %%%  
        end
        
        %% EM
        function obj = EM(obj,activeset)
            % E step (calculate posteriors with the active set)
            post = stageEM_posterior(obj.z,obj.C,obj.W,activeset);
            % M step
            if obj.Mstep_switch == 1
                % full M step
                [obj.C,obj.W] = stageEM_Mstep(obj.z,post);
            elseif obj.Mstep_switch == 2
                % partial M step
                [obj.C,obj.W] = stageEM_Mstep_partial(obj.z,obj.C,post,activeset);
            end
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
        %% log likelihood
        function output = loglikelihood(obj)
            output = stageEM_logL(obj.z,obj.W,obj.C,1:obj.m);
        end
    end
end