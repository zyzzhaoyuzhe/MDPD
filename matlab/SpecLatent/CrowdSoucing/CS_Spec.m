classdef CS_Spec < CrowdSourcing
    %% CrowdSourcing by Spectral Methods and EM
    % This method performs a two-stage method, based on Spectral Methods
    % Meet EM: A Provably Optimal Algorithm for Crowdsourcing by Yunchen
    % Zhang et.al.
    
    %%
    properties
        %% from class CrowdSourcing
        %{
        % Index i for workers, j for items (questions), l for labels
        m = 0 % # of workers
        n = 0 % # of items
        k = 0 % # of labels
        z %  k by n by m array. Use binary encoding for workers' answers (raw data)
        %}
        
        partition = cell(3,1) % the three way partition of workers
        Zg % k by n by 3 array. Group average answer.
        Cg % k by k by 3. Group average confusion matrix
    end

    %%
    methods
        %% constructor
        function obj = CS_Spec(m,n,k)
            obj@CrowdSourcing(m,n,k);
        end
        %% Learning the model from raw data z (binary encoding)
        function obj = learn(obj)
            obj.stage1;
            %%
            obj.stage2(1);
        end
        %% estimate confusion matrics
        function obj = stage1(obj)
            Wg = zeros(obj.c,obj.c,3);
            obj.get_partition;
            obj.get_GroupAve;
            perm = {[2,3,1],[3,1,2],[1,2,3]};
            for g = 1:3
                a = perm{g}(1);
                b = perm{g}(2);
                c = perm{g}(3);
                [M2,M3] = obj.get_Tensor(a,b,c);
                [Mu,W] = TensorPower(M2,M3,obj.c);
                % 
                obj.Cg(:,:,g) = Mu;
                Wg(:,:,g) = W;
            end
            obj.W = mean(Wg,3);
            % Use Cg to recover C
            for i = 1:obj.m
                tempC = zeros(obj.c,obj.c);
                for g = 1:3
                    if ~ismember(i,obj.partition{g})
                        tempC = tempC+obj.get_C_i(i,g);
                    end
                end
                tempC = tempC/2;
                obj.C(:,:,i) = tempC;
            end
        end
        %% Use EM to refine the stage 1 estimation
        function obj = stage2(obj,iter)
            for t = 1:iter
                q = obj.posterior;
                % Update Confusion Matrix
                obj.Mstep(q);
            end
        end
    end
    
    %% helper functions
    methods
        %% generate three way partition
        function obj = get_partition(obj)
            temp = floor(obj.m/3);
            idx = randperm(obj.m);
            obj.partition{1} = idx(1:temp);
            obj.partition{2} = idx(temp+1:2*temp);
            obj.partition{3} = idx(2*temp+1:end);
        end
        %% calculate group average answer
        function obj = get_GroupAve(obj)
            obj.Zg = zeros(obj.c,obj.n,3);
            for g = 1:3
                obj.Zg(:,:,g) = mean(obj.z(:,:,obj.partition{g}),3);
            end
        end
        %% get 2nd and 3rd order tensor
        function [M2,M3] = get_Tensor(obj,a,b,c)
            % a b c are permutation of 1 2 3
            Acb = 1/obj.n*obj.Zg(:,:,c)*obj.Zg(:,:,b)';
            Aab = 1/obj.n*obj.Zg(:,:,a)*obj.Zg(:,:,b)';
            Aca = 1/obj.n*obj.Zg(:,:,c)*obj.Zg(:,:,a)';
            Aba = 1/obj.n*obj.Zg(:,:,b)*obj.Zg(:,:,a)';
            % 
            Ba = Acb/Aab*obj.Zg(:,:,a);
            Bb = Aca/Aba*obj.Zg(:,:,b);
            M2(:,:,1) = Ba;
            M2(:,:,2) = Bb;
            M3(:,:,1) = Ba;
            M3(:,:,2) = Bb;
            M3(:,:,3) = obj.Zg(:,:,c);
        end
        %% Get C from Cg
        function C_i = get_C_i(obj,i,g)
            temp = 1/obj.n*obj.z(:,:,i)*obj.Zg(:,:,g)';
            C_i = temp/(obj.W*obj.Cg(:,:,g)');
            % if C_i has negative value set to zeros
            C_i(C_i<0.0001) = 0;
            % if I have all zero columns just set to uniform distribution
            foo = sum(C_i,1);
            C_i(:,foo==0) = 1/obj.c;
            % normalize C_i to make columns sum to one
            C_i = C_i*diag(1./sum(C_i,1));
        end
        %% E step for stage 2
        % use posterior function in parent class
        %% M step for stage 2
        function obj = Mstep(obj,q)
            % update confusion matrix
            for i = 1:obj.m
                temp = obj.z(:,:,i)*q';
                temp = bsxfun(@rdivide,temp,sum(temp,1));
                obj.C(:,:,i) = temp;
            end
            % update the distribution of labels
            temp = sum(q,2);
            obj.W = diag(temp/sum(temp));
        end
    end
end