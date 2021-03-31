from scipy import sparse
import numpy as np

DTYPE = np.float64
from numba import jit


# tri is simplices 2D int32 (x,y,z)
# @jit
def compute_block_periodic(tri, k, b):
    #
    # A = sparse(b*k,b*k);
    A = sparse.lil_matrix((b * k * 4, b * k * 4), dtype=DTYPE)  # The matlab code kept dynamically growing the matrix!

    #
    # % For convenience, first create the Laplacian matrix of the 3 by 3 block triangulation.
    # % Then use only the middle block to construct A.
    # B = sparse(9*k,9*k);
    B = sparse.lil_matrix((9 * k, 9 * k), dtype=DTYPE)

    #
    # % Make a standard log-normal random matrix of size k by 9*k.
    # % Only part of it will be used
    # R = -lognrnd(0,1,k,9*k);
    R = -np.random.lognormal(0., 1., size=(k, 9 * k))

    #

    # tri = sort(tri,2);
    tri = np.sort(tri, 1)

    # for i = 1:length(tri) % Run over all the triangles
    # 	t = tri(i,:); % For convenience, store triangle i in array t
    # 	B(t(1),t(2)) = R(mod(t(1)-1,k)+1,t(2)-t(1)); % Maintains periodicity in the middle block
    # 	B(t(1),t(3)) = R(mod(t(1)-1,k)+1,t(3)-t(1)); % Maintains periodicity in the middle block
    # 	B(t(2),t(3)) = R(mod(t(2)-1,k)+1,t(3)-t(2)); % Maintains periodicity in the middle block
    # 	% Symmetrize
    # 	B(t(2),t(1)) = B(t(1),t(2));
    # 	B(t(3),t(1)) = B(t(1),t(3));
    # 	B(t(3),t(2)) = B(t(2),t(3));
    # end
    for i in range(tri.shape[0]):
        t = tri[i, :]
        B[t[0], t[1]] = R[(t[0] - 1) % k, t[1] - t[0]]
        B[t[0], t[2]] = R[(t[0] - 1) % k, t[2] - t[0]]
        B[t[1], t[2]] = R[(t[1] - 1) % k, t[2] - t[1]]
        B[t[1], t[0]] = B[t[0], t[1]]
        B[t[2], t[0]] = B[t[0], t[2]]
        B[t[2], t[1]] = B[t[1], t[2]]

    #
    # % Plug the rows of B corresponding to the (2,2) block into A in the proper places,
    # % given that B has 3 by 3 blocks, while A has b by b blocks. This will
    # % define the (2,2) block of A
    #
    # A((b+1)*k+1:(b+2)*k,1:3*k) = B((3+1)*k+1:(3+2)*k,1:3*k);
    # A((b+1)*k+1:(b+2)*k,b*k+1:(b+3)*k) = B((3+1)*k+1:(3+2)*k,3*k+1:(3+3)*k);
    # A((b+1)*k+1:(b+2)*k,2*b*k+1:(2*b+3)*k) = B((3+1)*k+1:(3+2)*k,2*3*k+1:(2*3+3)*k);
    A_rows_range = slice((b + 1) * k, (b + 2) * k)
    B_rows_range = slice((3 + 1) * k, (3 + 2) * k)
    cols1 = slice(0, 3 * k)
    A[A_rows_range, cols1] = B[B_rows_range, cols1]
    cols2a = slice(b * k, (b + 3) * k)
    cols2b = slice(3 * k, (3 + 3) * k)
    A[A_rows_range, cols2a] = B[B_rows_range, cols2b]
    cols3a = slice(2 * b * k, (2 * b + 3) * k)
    cols3b = slice(2 * 3 * k, (2 * 3 + 3) * k)
    A[A_rows_range, cols3a] = B[B_rows_range, cols3b]

    #
    # % Now create the rest of the doubly periodic A from its (2,2) block.
    # for ib = 0:b^2-1 % Run over the blocks, starting from 0 for convenience.
    # 	             % ib = b+1 corresponds to block (2,2).
    for ib in range(0, b ** 2):
        # 	A(ib*k+1:ib*k+k,ib*k+1:ib*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+1)*k+1:(b+1)*k+k);
        dst_idxs = slice(ib * k, ib * k + k)
        src_idxs = slice((b + 1) * k, (b + 1) * k + k)
        A[dst_idxs, dst_idxs] = A[src_idxs, src_idxs]

        def conditional_fill(conditions_and_dst_cols_starts,  #: List[Tuple[Callable[[], bool], int]],
                             no_matches_dst_cols_start,  #: int,
                             src_cols_start):
            applied: bool = False
            src_cols = slice(src_cols_start, src_cols_start + k)
            dst_cols = []
            for option, dst_cols_start in conditions_and_dst_cols_starts:
                if option():
                    applied = True
                    dst_cols = slice(dst_cols_start, dst_cols_start + k)
                    break
            if not applied:
                dst_cols_start = no_matches_dst_cols_start
                dst_cols = slice(dst_cols_start, dst_cols_start + k)
            A[dst_idxs, dst_cols] = A[src_idxs, src_cols]

        # if (mod(ib,b) == 0) % North block
        # 		A(ib*k+1:ib*k+k,(ib-1+b)*k+1:(ib-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,b*k+1:b*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib-1)*k+1:(ib-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,b*k+1:b*k+k);
        # 	end
        conditional_fill([(lambda: ib % b == 0,
                           (ib - 1 + b) * k)],
                         (ib - 1) * k,
                         src_cols_start=b * k)

        # 	if (mod(ib,b) == b-1) % South block
        # 		A(ib*k+1:ib*k+k,(ib+1-b)*k+1:(ib+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+2)*k+1:(b+2)*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib+1)*k+1:(ib+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,(b+2)*k+1:(b+2)*k+k);
        # 	end
        conditional_fill([(lambda: ib % b == b - 1, (ib + 1 - b) * k)],
                         (ib + 1) * k,
                         src_cols_start=(b + 2) * k)

        # 	if (ib < b) % West block
        # 		A(ib*k+1:ib*k+k,(ib-b+b^2)*k+1:(ib-b+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,k+1:k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib-b)*k+1:(ib-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,k+1:k+k);
        # 	end
        conditional_fill([(lambda: ib < b,
                           (ib - b + b ** 2) * k)],
                         (ib - b) * k,
                         src_cols_start=k)

        # 	if (ib >= b^2-b) % East block
        # 		A(ib*k+1:ib*k+k,(ib+b-b^2)*k+1:(ib+b-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+1)*k+1:(2*b+1)*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib+b)*k+1:(ib+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+1)*k+1:(2*b+1)*k+k);
        # 	end
        conditional_fill([(lambda: ib >= b ** 2 - b,
                           (ib + b - b ** 2) * k)],
                         (ib + b) * k,
                         src_cols_start=(2 * b + 1) * k)

        # 	if (ib == 0) % NorthWest block
        # 		A(ib*k+1:ib*k+k,(b^2-1)*k+1:(b^2-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
        # 	elseif (mod(ib,b) == 0)
        # 		A(ib*k+1:ib*k+k,(ib-b-1+b)*k+1:(ib-b-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
        # 	elseif (ib < b)
        # 		A(ib*k+1:ib*k+k,(ib-b-1+b^2)*k+1:(ib-b-1+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib-b-1)*k+1:(ib-b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,1:k);
        # 	end
        conditional_fill(
            [(lambda: ib == 0, (b ** 2 - 1) * k),
             (lambda: ib % b == 0, (ib - b - 1 + b) * k),
             (lambda: ib < b, (ib - b - 1 + b ** 2) * k)],
            (ib - b - 1) * k,
            0
        )

        # 	if (ib == b-1) % SouthWest block
        # 		A(ib*k+1:ib*k+k,(b^2-b)*k+1:(b^2-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
        # 	elseif (mod(ib,b) == b-1)
        # 		A(ib*k+1:ib*k+k,(ib-b+1-b)*k+1:(ib-b+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
        # 	elseif (ib < b)
        # 		A(ib*k+1:ib*k+k,(ib-b+1+b^2)*k+1:(ib-b+1+b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib-b+1)*k+1:(ib-b+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*k+1:2*k+k);
        # 	end
        conditional_fill(
            [(lambda: ib == b - 1, (b ** 2 - b) * k),
             (lambda: ib % b == b - 1, (ib - b + 1 - b) * k),
             (lambda: ib < b, (ib - b + 1 + b ** 2) * k)],
            (ib - b + 1) * k,
            2 * k
        )

        # 	if (ib == b^2-b) % NorthEast block
        # 		A(ib*k+1:ib*k+k,(b-1)*k+1:(b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
        # 	elseif (mod(ib,b) == 0)
        # 		A(ib*k+1:ib*k+k,(ib+b-1+b)*k+1:(ib+b-1+b)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
        # 	elseif (ib >= b^2-b)
        # 		A(ib*k+1:ib*k+k,(ib+b-1-b^2)*k+1:(ib+b-1-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib+b-1)*k+1:(ib+b-1)*k+k) = A((b+1)*k+1:(b+1)*k+k,2*b*k+1:2*b*k+k);
        # 	end
        conditional_fill(
            [(lambda: ib == b ** 2 - b, (b - 1) * k),
             (lambda: ib % b == 0, (ib + b - 1 + b) * k),
             (lambda: ib >= b ^ 2 - b, (ib + b - 1 - b ** 2) * k)],
            (ib + b - 1) * k,
            2 * b * k
        )

        # 	if (ib == b^2-1) % SouthEast block
        # 		A(ib*k+1:ib*k+k,1:k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
        # 	elseif (mod(ib,b) == b-1)
        # 		A(ib*k+1:ib*k+k,(ib+b+1-b)*k+1:(ib+b+1-b)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
        # 	elseif (ib >= b^2-b)
        # 		A(ib*k+1:ib*k+k,(ib+b+1-b^2)*k+1:(ib+b+1-b^2)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
        # 	else
        # 		A(ib*k+1:ib*k+k,(ib+b+1)*k+1:(ib+b+1)*k+k) = A((b+1)*k+1:(b+1)*k+k,(2*b+2)*k+1:(2*b+2)*k+k);
        # 	end
        #
        # end
        conditional_fill(
            [(lambda: ib == b ** 2 - 1, (b + 1) * k),
             (lambda: ib % b == b - 1, (ib + b + 1 - b) * k),
             (lambda: ib >= b ** 2 - b, (ib + b + 1 - b ** 2) * k)],
            (ib + b + 1) * k,
            (2 * b + 2) * k
        )

    #
    # % Zerosum
    # for i = 1:length(A)
    # 	A(i,i) = -sum(A(i,:));
    # end
    for i in range(A.shape[0]):
        A[i, i] = -np.sum(A[i, :])

        #
        # % Python Matlab engine does not support sparse arrays BUT OCTAVE DOES!
        # %A = full(A);
        # end
    return A.tocsr()
