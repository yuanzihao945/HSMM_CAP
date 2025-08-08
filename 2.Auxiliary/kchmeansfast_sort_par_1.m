function a = kchmeansfast_sort_par_1(X, w, K)
[~, P, N] = size(X);
a = zeros(1, P);
parfor iN = 1 : N
    State_iN = kmeans(X(:, w==1, iN), K);
    [~, a_iN] = CA_SVS_epsilon_part(X(:, :, iN), State_iN, K);
    a = a + a_iN;
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
