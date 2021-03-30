function [rows, cols] = square_P(P_rows, P_cols, P_values, total_size, coarse_nodes)
total_size = double(total_size);
P_num_rows = total_size;
[~, P_num_cols] = size(coarse_nodes);

P = sparse(P_rows, P_cols, P_values, P_num_rows, P_num_cols);
P_square = sparse(total_size, total_size);
P_square(:, coarse_nodes) = P;
[rows, cols] = find(P_square);
end

% Find out how many rows and cols our matrix will have
%% num_rows is total_size
%% num_cols is the number of columns in coarse_nodes
% Create a sparse matrix P of size (num_rows,num_cols).
%% row indices are in P_rows
%% col indicides are in P_cols
%% values are in P_values
% Create a sparse matrix P_square of num_rows x num_rows
% Set P_square's coarse nodes locations to P's vals
%% Return the locations of non-zero elems in P_square (sparsity pattern)
