function [s,v] = TensorPower_helper(M3,L,N,lambda,v,l)
% helper function. Use robust tensor power method to return the first
% eigenpair and deflated tensor Res
dim = size(M3,1);
n = size(M3,2);

Lambda = zeros(L,1);
Theta = zeros(dim,L);
for tau = 1:L
    theta = rand(dim,1);
    theta = theta/norm(theta);
    for t = 1:N
        temp2 = theta'*M3(:,:,2);
        temp3 = theta'*M3(:,:,3);
        temp = temp2.*temp3;
        theta_pre = theta;
        
        theta = 1/n*M3(:,:,1)*temp';
        for k = 1:l-1
            theta = theta-lambda(k)*v(:,k)*(theta_pre'*v(:,k))^2;
        end
        theta = theta/norm(theta);
    end
    Theta(:,tau) = theta;
    Lambda(tau) = mean((theta'*M3(:,:,1)).*(theta'*M3(:,:,2)).*(theta'*M3(:,:,3)));
end
[~,idx] = max(Lambda);
theta = Theta(:,idx);
for t = 1:N
    temp2 = theta'*M3(:,:,2);
        temp3 = theta'*M3(:,:,3);
        temp = temp2.*temp3;
        theta_pre = theta;
        
        theta = 1/n*M3(:,:,1)*temp';
        for k = 1:l-1
            theta = theta-lambda(k)*v(:,k)*(theta_pre'*v(:,k))^2;
        end
        theta = theta/norm(theta);
end
v = theta;
s = mean((v'*M3(:,:,1)).*(v'*M3(:,:,2)).*(v'*M3(:,:,3)));
end