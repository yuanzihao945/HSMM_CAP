% Author: ZH.Yuan
% Update: 2025/08/08
% Email: zihaoyuan@whut.edu.cn

clc;
clearvars;
close all;

%% Generation process
% Rand Samples
N = 50;                         % sample number
P = 1000;                       % feature number
T = 210;                        % time number
q = 50;                         % number of active variable
K = 8;                          % cluster number
% Transition Probability Matrix
ProbM = [0.15  0.50  0.15  0.20  0.00  0.00  0.00  0.00 ; ...
         0.25  0.15  0.40  0.10  0.00  0.10  0.00  0.00 ; ...
         0.60  0.10  0.10  0.10  0.10  0.00  0.00  0.00 ; ...
         0.25  0.30  0.20  0.25  0.00  0.00  0.00  0.00 ; ...
         0.00  0.00  0.00  0.00  0.25  0.30  0.20  0.25 ; ...
         0.00  0.15  0.00  0.00  0.15  0.10  0.50  0.10 ; ...
         0.00  0.00  0.15  0.00  0.05  0.40  0.15  0.25 ; ...
         0.00  0.00  0.00  0.00  0.05  0.10  0.55  0.30];
mc = dtmc(ProbM);

figure;
imagesc(ProbM);
colormap(flip(othercolor('RdYlBu7'), 1))
axis square;
colorbar;
title('True Transition Probability Matrix');
textStrings = strtrim(cellstr(num2str(ProbM(:), '%0.2f')));
[Wx, Wy] = meshgrid(1 : K);
hStrings = text(Wx(:), Wy(:), textStrings(:), 'HorizontalAlignment', 'center');

% Intial correlation parameters
mu = zeros(K, P);
for k = 1 : K
    mu(k, q * (k - 1) + (1 : q)) = 1.5;
end

% Sample Random State Series by Transition Probability Matrix
TState = zeros(N, T);
X = zeros(T, P, N);
for i = 1 : N
    TState(i, :) = simulate(mc, T - 1);
    X(:, :, i) = 0.9 * (mu(TState(i, :), :) + randn(T, P)) + 0.1 * trnd(1, T, P);
end
s = 300;
sk = 50;


%% Clsuter and Screen
% Feature screening
tic
[EState, TPM, W] = HSMM_CAP(X, K, sk, s);
toc
