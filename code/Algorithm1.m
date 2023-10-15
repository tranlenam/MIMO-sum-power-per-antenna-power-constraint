function [lambdaSPC,lambdaPAPC,gamma] = Algorithm1(phi,SPC,PAPC)
% Algorithm1 solves (7) in the paper
%   Detailed explanation goes here
N = length(phi);
P_hat = sum(PAPC);
P = SPC + P_hat;
P0_tilde = SPC-P_hat; % should be less than ZERO

% the following code is to solve (9) using CVX,
%{
cvx_begin quiet
variable lambdaSPC_cvx  ;
variable w_cvx(N) ;
dual variable gamma_cvx; % dual variable for (9b)
dual variable mu0_cvx;
dual variable mu_cvx; % dual variable for (9c)
obj = (w_cvx)'*phi - sum(log(w_cvx));
minimize(obj)
subject to
gamma_cvx: (w_cvx'*PAPC + lambdaSPC_cvx*P0_tilde)  == P;
mu0_cvx : lambdaSPC_cvx >= 0;
mu_cvx : (w_cvx-lambdaSPC_cvx) >= 0;
cvx_end
lambdaPAPC_cvx  = w_cvx-lambdaSPC_cvx;
%}

% check trivial case 1
w = 1./phi;
lambdaSPC = (P-w'*PAPC)/P0_tilde;

if (lambdaSPC > 0) && (min(w)>lambdaSPC)
    lambdaPAPC = w-lambdaSPC;
    gamma = 0;
    return
end
% trivial case 2 w1=w2=..=\lamda0
lambdaSPC = P/(P0_tilde+P_hat); 
w = lambdaSPC*ones(N,1);
gamma = sum(1./w-phi)/(P0_tilde+P_hat);
mu = phi - 1./w + gamma*PAPC;

if(min(mu)>=0)
    lambdaPAPC = zeros(N,1);
    return
end

% other cases
gamma_min = 0;
lambdaPAPC = zeros(N,1);
gamma_max = N/P - min(phi)/max(PAPC);
w =  zeros(N,1);
while((gamma_max-gamma_min) > 1e-6)
    phi_temp =  phi;
    PAPC_temp = PAPC;
    gamma_bar = (gamma_min+gamma_max)/2;

    % step 3
    [~, ind] = sort(1./(phi_temp + gamma_bar*PAPC_temp),'descend');
    phi_temp = phi_temp(ind);
    PAPC_temp = PAPC_temp(ind);

    % step 4
    for k=N-1:-1:1
        if (1/(phi_temp(k) + gamma_bar*PAPC_temp(k)) > ...
                ((N-k)/(sum(phi_temp(k+1:N)) + gamma_bar*(P0_tilde + sum(PAPC_temp(k+1:N))))))
            break
        end
    end
    %k
    % step 5
    w(1:k) = 1./(phi_temp(1:k) + gamma_bar*PAPC_temp(1:k));
    lambdaSPC = (N-k)/(sum(phi_temp(k+1:N)) + gamma_bar*(P0_tilde+sum(PAPC_temp(k+1:N))));
    w(k+1:N) = lambdaSPC;

    % step 6
    if (w'*PAPC_temp + P0_tilde*lambdaSPC - P) > 0
        gamma_min = gamma_bar;
    else
        gamma_max = gamma_bar;
    end
end

lambdaPAPC(ind) = w - lambdaSPC;
gamma = gamma_bar;
end

