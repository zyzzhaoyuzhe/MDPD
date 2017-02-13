%% calculate Co-occurance matrices from raw data
% INPUT:
% z is the raw data (r by n by m);
% post: posterior distribution of the current model (c by n)
% W_fs: contains the update of prior based on current model. f_S(y=k)
% OUTPUT:
% ratio: f_S(x_i_x_j|y=k)/(f_S(x_i|y=k)f_S(x_j|y=k)[def: up/down]
% (m by m by r by r by c) 

function ratio = stageEM_ratio(z,post,W_fs,C_fs)
c = size(W_fs,1);
r = size(z,1);
n = size(z,2);
m = size(z,3);

ratio = zeros(m,m,r,r,c);

for k = 1:c
    % adding weights to samples (up)
    z_tmp = z;
    post_tmp = sqrt(post(k,:));
    for ll = 1:r
        foo = z_tmp(ll,:,:);
        foo = squeeze(foo);
        foo = bsxfun(@times,foo,post_tmp');
        z_tmp(ll,:,:) = foo;
    end
    
    % prepare C_fs (down)
    C_tmp = C_fs(:,k,:);
    C_tmp = squeeze(C_tmp);
    
    for l1 = 1:r
        % (up)
        z1 = z_tmp(l1,:,:);
        z1 = shiftdim(z1);
        % (down)
        c1 = C_tmp(l1,:);
        
        for l2 = 1:r
            % (up)
            z2 = z_tmp(l2,:,:);
            z2 = shiftdim(z2);
            % (down)
            c2 = C_tmp(l2,:);
            % matrix (up)
            Mup = 1/n*z1'*z2;
            % matrix (down)
            Mdown = c1'*c2;
            
            foo = Mup./Mdown;
            
%             foo = Mup;
            
            foo(isnan(foo)) = 0;
            ratio(:,:,l1,l2,k) = foo;
        end
    end
    ratio(:,:,:,:,k) = ratio(:,:,:,:,k)/W_fs(k,k);
end
end
