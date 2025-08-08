function [State, a] = kchmeansfast_sort_par_2(X, w, State, MaxIter, Prn)
[P, N] = size(X, [2 3]);
K = size(w, 1);
a = zeros(K, P);
parfor iN = 1 : N
    State(iN, :) = kchmeansfast_sort_2(X(:, :, iN), w, State(iN, :)', MaxIter);
    a = a + Prn(iN) * CA_SVS_epsilon_part(X(:, :, iN), State(iN, :)', K);
end
end


function [idx, ccop, distop, iterop] = kchmeansfast_sort_2(data, PkM, idx0, MaxIter)
[N, P] = size(data);
K = size(PkM, 1);
idx = idx0;
iterop = 1;
mdata = mean(data);
ccop = zeros(K, P);
distop = zeros(N, K);
while iterop <= MaxIter
    ccop = repmat(mdata, K, 1);
    for iop = 1 : K
        ccop(iop, PkM(iop, :)==1) = mean(data(idx == iop, PkM(iop, :)==1), 1);
    end
    distop = zeros(N, K);
    for iop = 1 : K
        distop(:, iop) = sum((data - ccop(iop, :)).^2, 2);
    end
    [~, idx_new] = min(distop, [], 2);
    if idx_new == idx
        break
    else
        iterop = iterop + 1;
    end
    idx = idx_new;
end
end


function [upsilonk_hat, upsilon_hat] = CA_SVS_epsilon_part(X, Y, K)
[N, P] = size(X);
pk_hati = zeros(K, 1);
tauk_hat = zeros(K, P);
upsilonk_hat = zeros(K, P);
for k = 1 : K
    pk_hati(k) = sum(Y == k) / N;
    tauk_hat(k, :) = 1 / (N * (N+1) * pk_hati(k)) * sum(X(Y == k, :), 1) - 0.5;
    upsilonk_hat(k, :) = 12 * (N+1) * pk_hati(k) / (1 - pk_hati(k)) * tauk_hat(k, :).^2;
end
upsilonk_hat(isnan(upsilonk_hat)) = 0;
upsilon_hat = (length(unique(Y)) - 1) / length(unique(Y)) * sum(upsilonk_hat);
end
