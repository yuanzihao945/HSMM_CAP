function [chain_prob, chain_prob_] = gam_chain_prob(state, du, param)

K = length(param);

N = length(state);
chain_prob_ = zeros(1, N);

for k = 1 : K
    chain_prob_(state == k) = gampdf(du(state == k), 1, param(k));
end

chain_prob = prod(chain_prob_);
