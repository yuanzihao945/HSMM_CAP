function [idx, ccop, distop] = kchmeansfast_sort(data, PkM, idx0, MaxIter, display, dd, MaxMemory)
% KCHMEANS - Improved kmeans clustering algorithm for clustering-heterogeneous data
%
% Input :
%   data    -  Unsupervised data requiring clustering
%   PkM     -  Heterogeneity Category Matrix (heterogeneity)
%   idx0    -  Initial idx
%   MaxIter -  Maximum number of iterations of the algorithm
%   display -  Show(display = 1) or not show(display = 0) the maximum
%               number of iterations at the end of the algorithm
%   dd      -  Norm number
%
% Output:
%   idx     -  Label of clustering results
%   ccop    -  Center of clustering
%   distop  -  Distance from sample to center of each cluster
%
% Usage:
%   [idx, ccop, distop] = KCHMEANS(data, Pk, MaxIter, method, display)
%           'MaxIter' has default value 100;
%           'method' has default set 'randperm';
%           'display' has default value 0;
%
% See also KCHMEANS_INTER, KCHNNPRE, KCHMEANSFAST, KCHMEANSFASTS

% Author  : ZH.Yuan
% Update  : 2025/08/08
% Email   : zihaoyuan@whut.edu.cn (If any suggestions or questions)


if ~exist('MaxMemory', 'var') || isempty(MaxMemory)
    MaxMemory = 16;
end
if ~exist('MaxIter', 'var') || isempty(MaxIter)
    MaxIter = 100;       % Maximum number of iterations
end
if ~exist('display', 'var') || isempty(display)
    display = 0;         % The number of iterations is not displayed by default
end
if ~exist('dd', 'var') || isempty(dd)
    dd = 2;              % The norm of distance by default
end

if (numel(data) * size(PkM, 1) * 8 / 1024^3) > (MaxMemory * 0.9)

    [idx, ccop, distop] = kchmeansfast_sort_o(data, PkM, idx0, MaxIter, display, dd);
    
else

    if dd == 2
        [idx, ccop, distop, iterop] = kchmeansfast_sort_2_mex(data, PkM, idx0, MaxIter);
    end

    if dd ~= 2
        [N, ~] = size(data);     % N is the number of data and P is the dimension
        K = size(PkM, 1);
        idx = idx0;

        % Iterative update
        iterop = 1;
        mdata = mean(data);
        while iterop <= MaxIter

            ccop = repmat(mdata, K, 1);
            for iop = 1 : K
                ccop(iop, logical(PkM(iop, :))) = mean(data(idx == iop, logical(PkM(iop, :))));
            end

            distop = zeros(N, K);
            for iop = 1 : K
                distop(:, iop) = sum(abs(data - ccop(iop, :)).^dd, 2);
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

    if iterop > MaxIter
        iterop = MaxIter;
    end

    if display
        fprintf([mfilename ' stop at the %d-th iteration\n'], iterop); % Output iterations
    end

end

end


function [idx, ccop, distop] = kchmeansfast_sort_o(data, PkM, idx0, MaxIter, display, dd)
% KCHMEANS - Improved kmeans clustering algorithm for clustering-heterogeneous data
%
% Input :
%   data    -  Unsupervised data requiring clustering
%   PkM     -  Heterogeneity Category Matrix (heterogeneity)
%   idx0    -  Initial idx
%   MaxIter -  Maximum number of iterations of the algorithm
%   display -  Show(display = 1) or not show(display = 0) the maximum
%               number of iterations at the end of the algorithm
%   dd      -  Norm number
%
% Output:
%   idx     -  Label of clustering results
%   ccop    -  Center of clustering
%   distop  -  Distance from sample to center of each cluster
%
% Usage:
%   [idx, ccop, distop] = KCHMEANS(data, Pk, MaxIter, method, display)
%           'MaxIter' has default value 100;
%           'method' has default set 'randperm';
%           'display' has default value 0;
%
% See also KCHMEANS_INTER, KCHNNPRE, KCHMEANSFAST, KCHMEANSFASTS

% Author  : ZH.Yuan
% Update  : 2021/08/31 (First Version: 2022/03/16)
% Email   : zihaoyuan@whut.edu.cn (If any suggestions or questions)


if ~exist('MaxIter', 'var') || isempty(MaxIter)
    MaxIter = 100;       % Maximum number of iterations
end
if ~exist('display', 'var') || isempty(display)
    display = 0;         % The number of iterations is not displayed by default
end
if ~exist('dd', 'var') || isempty(dd)
    dd = 2;              % The norm of distance by default
end

[N, ~] = size(data);     % N is the number of data and P is the dimension
K = size(PkM, 1);
idx = idx0;


% Iterative update
iterop = 1;
mdata = mean(data);
while iterop <= MaxIter

    ccop = repmat(mdata, K, 1);
    for iop = 1 : K
        ccop(iop, logical(PkM(iop, :))) = mean(data(idx == iop, logical(PkM(iop, :))));
    end

    distop = zeros(N, K);
    switch dd
        case 2
            for iop = 1 : K
                distop(:, iop) = sum((data - ccop(iop, :)).^2, 2);
            end
        otherwise
            for iop = 1 : K
                distop(:, iop) = sum(abs(data - ccop(iop, :)).^dd, 2);
            end
    end
    [~, idx_new] = min(distop, [], 2);

    if idx_new == idx
        break
    else
        iterop = iterop + 1;
    end
    idx = idx_new;

end

if iterop > MaxIter
    iterop = MaxIter;
end

if display
    fprintf([mfilename ' stop at the %d-th iteration\n'], iterop); % Output iterations
end

end







