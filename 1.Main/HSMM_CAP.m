function [State, TPM, w, du, State_Prob] = HSMM_CAP(X, K, sk, s, display, MaxIter, WM)
% Co-Activation Pattern Analysis based on Hidden Semi-Markov Model for
% Brain Spatiotemporal Dynamics
%
% Input :
%   X       -  Data Matrix
%   K       -  Total number of Clusters
%   sk      -  The number of co-activated voxels in each state
%   s       -  The number of co-activated voxels in all states
%   display -  Show(display = 1) or not show(display = 0) the maximum 
%               number of iterations at the end of the algorithm
%   MaxIter -  Maximum number of iterations of the algorithm 
%   WM      -  HSMM-weight methods
%
% Output:  
%   State   -  state of clustering results
%   TPM     -  Transition Probability Matrix
%   w       -  Center of clustering
%   du      -  duration of HSMM
%   State_Prob
%           -  Probablity of each state
%
% Usage:
%   [State, TPM, w] = HSMM_CAP(X, K, sk, s, display, MaxIter, WM)
%           'sk' The number of voxels in each state
%           'MaxIter' has default value 500;
%           'display' has default value 0;
%
% Author  : ZH.Yuan
% Update  : 2025/08/08
% Email   : zihaoyuan@whut.edu.cn (If any suggestions or questions)


[T, P, N] = size(X);           % size of sample matrix

%%
if ~exist('sk','var') || isempty(sk)
    sk = min([round(T / log(T)) ceil(P / K)]); % given the number screening last
end
if ~exist('s','var') || isempty(s)
    s = min([K * sk, ceil(0.6 * P)]); % given the number screening last
end
if ~exist('MaxIter', 'var') || isempty(MaxIter)
    MaxIter = 100;      % Maximum number of iterations
end
if ~exist('display', 'var') || isempty(display)
    display = 0;        % The number of iterations is not displayed by default
end
if ~exist('WM', 'var') || isempty(WM)
    WM = 'MW';
end

if isscalar(MaxIter)
    MaxIter = MaxIter * ones(1, 3);
end
MaxIter1 = MaxIter(1);
MaxIter2 = MaxIter(2);
MaxIter3 = MaxIter(3);

[~, Xplace] = sort(X, 1, 'descend');
[~, Xm] = sort(Xplace);
clear Xplace X


%% Process I
% StepI-1 Set the intial weight vector
w = ones(1, P);
iter1 = 0;

while iter1 < MaxIter1
    iter1 = iter1 + 1;
    w_old = w;

    a0 = kchmeansfast_sort_par_1_mex(Xm, w_old, K);
    clear mex

    a0k = maxk(a0, s);
    w(a0 >= a0k(end)) = 1;
    w(a0 < a0k(end)) = 0;

    % StepI-4 Judge whether convergence
    diff_rate0 = sum(abs(w - w_old)) / sum(abs(w_old));
    if diff_rate0 < 1e-3
        break
    end
end

State = reshape(kmeans(reshape(permute(Xm(:, w ~= 0, :), [1 3 2]), ...
            T * N, sum(w ~= 0)), K), T, N)';


%% Process II
w = repmat(w, K, 1);
iter2 = 0;

while iter2 < MaxIter2
    iter2 = iter2 + 1;
    w_old  = w;

    [State, a] = kchmeansfast_sort_par_2_mex(Xm, w, State, 100, ones(1, N)/N);
    clear mex

    [~, ap] = maxk(a, sk, 2);
    w = zeros(K, P);
    for k = 1 : K
        w(k, ap(k, :)) = 1;
    end

    [du, TPM, ~, ~, ~] = TPMatrix_HSMM(State, K);

    diff_rate = sum(abs(w - w_old), "all") / sum(w, 'all');
    if diff_rate < 1e-4
        break
    end

end


%% Process III
iter3 = 0;

while iter3 < MaxIter3
    iter3 = iter3 + 1;
    w_old = w;

    TPM_old = TPM;
    
    Prn = zeros(1, N);
    switch WM
        case 'PW'
            for iN = 1 : N
                change_point = find(ischange(State(iN, :), 'Threshold', 0.1) == 1);
                du_iN = diff([1, change_point, T+1]);
                state_iN = State(iN, [1, change_point]);
                Prn(iN) = sum(log(TPM_old(state_iN(1 : (end-1)) + K * (state_iN(2 : end) - 1))) + ...
                    log(gam_chain_prob(state_iN, du_iN, du)));
            end
            Prn = 1./(Prn);
        case 'IPW'
            for iN = 1 : N
                change_point = find(ischange(State(iN, :), 'Threshold', 0.1) == 1);
                du_iN = diff([1, change_point, T+1]);
                state_iN = State(iN, [1, change_point]);
                Prn(iN) =  - sum(log(TPM_old(state_iN(1 : (end-1)) + K * (state_iN(2 : end) - 1)))) - ...
                    log(gam_chain_prob(state_iN, du_iN, du));
            end
        case 'MW'
            for iN = 1 : N
                change_point = find(ischange(State(iN, :), 'Threshold', 0.1) == 1);
                du_iN = diff([1, change_point, T+1]);
                state_iN = State(iN, [1, change_point]);
                [~, chain_prob_] = gam_chain_prob(state_iN, du_iN, du);
                Prn(iN) = exp(mean(log(TPM_old(state_iN(1 : (end-1)) + K * (state_iN(2 : end) - 1)) ))) * ...
                    exp(mean(log(chain_prob_)));
            end
        case 'IPMW'
            for iN = 1 : N
                change_point = find(ischange(State(iN, :), 'Threshold', 0.1) == 1);
                du_iN = diff([1, change_point, T+1]);
                state_iN = State(iN, [1, change_point]);
                Prn(iN) = 1 ./ exp(mean(log(TPM_old(State(iN, 1 : (T-1)) + K * (State(iN, 2 : T) - 1)))));
            end
        case 'IM'
            for iN = 1 : N
                Prn(iN) = 1;
            end
    end
    

    Prn = Prn / sum(Prn);
    [State, a] = kchmeansfast_sort_par_2_mex(Xm, w, State, 100, Prn);

    [~, ap] = maxk(a, sk, 2);
    w = zeros(K, P);
    for k = 1 : K
        w(k, ap(k, :)) = 1;
    end

    [du, TPM, ~, ~, State_Prob] = TPMatrix_HSMM(State, K);

    diff_rate1 = sum(abs(TPM - TPM_old), "all") / K;
    diff_rate2 = sum(abs(w - w_old), "all") / sum(abs(w), "all");
    if diff_rate1 < 1e-4 && diff_rate2 < 1e-4
        break
    end

end


%%
if display  % Output iterations
    fprintf(['Part I of ' mfilename ': Initial value is stop at' ...
        ' the %d-th iteration\n'], iter1);
    fprintf(['Part II of ' mfilename ': CAP-process' ...
        ' is stop at the %d-th iteration\n'], iter2);
    fprintf(['Part III of ' mfilename ': HSMM-CAP process' ...
        ' is stop at the %d-th iteration\n'], iter3);
end


end
