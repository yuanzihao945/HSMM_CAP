function [du, P, T, du_all, State_Prob] = TPMatrix_HSMM(c, K)
% TPMatrix - Compute the State transition matrix T and State transition
% probability matrix P
% 
% Input :
%   c   -   State Matrix
%
% Output:
%   T   -   State transition matrix
%   P   -   State transition probability matrix
% 
% Author  : ZH.Yuan
% Update  : 2025/08/08
% Email   : zihaoyuan@whut.edu.cn (If any suggestions or questions)

if ~exist('K','var') || isempty(K)
    K = max(unique(c));           % dimension (number) of all state
end

Tn = zeros(K, K, size(c, 1));       % Pre State transition matrix T
du = zeros(K, 1);

for i = 1 : size(c, 1)
    change_point = find(ischange(c(i, :), 'Threshold', 0.1) == 1);
    du_i = diff([1, change_point, size(c, 2)+1]);
    state = c(i, [1, change_point]);
    for k1 = 1 : K
        loc_k1 = find(state == k1);     % Where the state i appears in observed data
        du_all{k1, i} = du_i(loc_k1);
        for k2 = 1 : K
            loc_k2 = find(state == k2); % Where the state j appears in observed data
            Tn(k1, k2, i) = sum(ismember(loc_k1 + 1 ,loc_k2)); 
        end
    end
end

for k = 1 : K
    du_k_all = [];
    for i = 1 : size(c, 1)
        du_k_all = [du_k_all du_all{k, i}];
    end
    du(k, :) = mean(du_k_all);
end

T = sum(Tn, 3);
P = T ./ repmat(sum(T, 2), 1, K);
for k = 1 : K
    if isnan(P(k, k))
        P(k, :) = zeros(1, K);
        P(k, k) = 1;
    end
end


State_Prob = zeros(K, size(c, 2), size(c, 1));
lambda = 1./ du;

for i = 1 : size(c, 1)
    State_Prob(c(i, 1), 1, i) = 1;
    for t = 2 : size(c, 2)
        for k1 = 1 : K
            for k2 = 1 : K
                if k1 == k2
                    State_Prob(k1, t, i) = State_Prob(k1, t, i) + State_Prob(k2, t-1, i) * exp(-lambda(k2));
                else
                    State_Prob(k1, t, i) = State_Prob(k1, t, i) + State_Prob(k2, t-1, i) * (1 - exp(-lambda(k2))) * P(k2, k1);
                end
            end
        end
    end
end

