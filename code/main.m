% channel
clear
clc
rng(1)
M = 2;
N = 4 ;
H = (randn(M,N) + 1i*randn(M,N))/sqrt(2);

% channel for Fig 2
%{
H = [0.1189+1i*0.1515 0.1238+1i*0.3326 0.8572+1i*0.1131;...
-0.3198-1i*0.3663 -0.6491+1i*0.2784 0.3392-1i*0.1974;...
-0.1019+ 1i*0.6639 0.3663-1i*0.3097 -0.1116-1i*0.1101];

[M, N] = size(H);
%}

P_hat = 0; % dbW, sum of per antenna power constraints
P_hat = 10^(P_hat/10); % to linear scale

PAPC = P_hat/N*ones(N,1); % per antenna power constraint
SPC = 0.6*P_hat; % sum power constraint

P = SPC + sum(PAPC);
%% solve (2) using CVX for comparison
% cvx_solver mosek % uncomment this line if you have CVX installed
cvx_expert true
cvx_begin quiet
variable X(N,N) complex semidefinite
maximize(log_det(eye(M)+H*X*H'))
subject to
real(trace(X)) <= SPC
diag(X) <= PAPC
X == hermitian_semidefinite(N)
cvx_end
optimal_obj_cvx = cvx_optval;


lambdaSPC = 1; % Lagrange multiplier for sum power constraint
lambdaPAPC = ones(N,1); % Lagrange multiplier for per antenna power constraint
maxIters  = 10;
objseq = zeros(maxIters,1);

%% Proposed CCP-like algorithm
for iIter=1:maxIters
    % solve (5) to update X_bar
    X_bar = MIMOcapacity(diag((lambdaSPC+lambdaPAPC).^(-0.5))*(H'),P);
    Phi = diag(lambdaSPC+lambdaPAPC) + H'*X_bar*H; % defined below (6)
    phi = real(diag(inv(Phi)));

    objseq(iIter) = real(log(det(H'*X_bar*H+diag(lambdaSPC+lambdaPAPC)))...
                        - sum(log(lambdaSPC+lambdaPAPC)));

    % solve (9) to update \lambda
    [lambdaSPC,lambdaPAPC,gamma] = Algorithm1(phi,SPC,PAPC);
end
%% plot residual error
semilogy(abs(optimal_obj_cvx-objseq))
xlabel('Iteration count')
ylabel('Residual error')
saveas(gcf,'../results/convergence.png')