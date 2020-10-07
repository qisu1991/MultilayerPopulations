using Printf
using Statistics
using LinearAlgebra
using DelimitedFiles
using IterativeSolvers
using SparseArrays
using TickTock

# rearrange the edge sequence in the increasing order of layer id and starting node id
function edge_seq_rearrangement(edge_seq::Array{Int64,2})
    layer_list = setdiff(edge_seq[:,4],[])
    L = maximum(layer_list)
    if length(setdiff([i for i = 1:L],layer_list)) != 0
        println("please label layers in the order 1, 2, ..., L")
    end
    node_list = setdiff(edge_seq[:,1],[])
    N = maximum(edge_seq[:,1])
    if length(setdiff([i for i = 1:N],node_list)) != 0
        println("please label nodes in the order 1, 2, ..., N")
    end

    edge_seq_rearranged = zeros(Int64,size(edge_seq))
    id = 1
    for ell in layer_list
        edge_list_ell = findall(isequal(ell),edge_seq[:,4])
        M = sparse(edge_seq[edge_list_ell,1],edge_seq[edge_list_ell,2],edge_seq[edge_list_ell,3])
        inds = findall(!iszero,M)
        a = getindex.(inds, 1)
        b = getindex.(inds, 2)
        edge_seq_rearranged[id:id+length(a)-1,1] = b
        edge_seq_rearranged[id:id+length(a)-1,2] = a
        edge_seq_rearranged[id:id+length(a)-1,3] = M[inds]
        edge_seq_rearranged[id:id+length(a)-1,4] = ell*ones(Int64,length(a))
        id = id+length(a)
    end

    return edge_seq_rearranged
end

# solve a set of equations to obtain beta_i, beta_ij, and gamma_ij, for any number of layers
# this function is used for rhoA > (rhoA)^o in layer one under a given strategy configuration
# input---
# multilayer network in the form of an edge sequanece: edge_seq [node id, node id, edge weight, layer id], where edge weights here are set to be any positive integer
# initial strategy configuration: xi_seq [node id, strategy, layer id]
# output---
# the first column of "solution" gives beta_ij
# the other columns of "solution" gives gamma_ij, where the ll_th column is the strategy assortment between layer 1 and layter ll
function beta_gamma(edge_seq::Array{Int64,2}, xi_seq::Array{Int64,2})
    N = maximum(edge_seq[:,1])
    L = maximum(edge_seq[:,4])
    M = spzeros(Float64,N*L,N)
    for ell = 1:L
        edge_list_ell = findall(isequal(ell),edge_seq[:,4])
        M_ell = sparse(edge_seq[edge_list_ell,1],edge_seq[edge_list_ell,2],edge_seq[edge_list_ell,3],N,N)
        M[(ell-1)*N+1:ell*N,:] = M_ell
    end
    pi = spzeros(Float64,L*N)
    for ell = 1:L
        pi_2 = sum(M[(ell-1)*N+1:ell*N,:], dims = 2)
        pi_2 = pi_2[:,1]
        pi_transient = spzeros(Float64, N)
        presence = findall(!iszero,pi_2)
        pi_transient[presence] = pi_2[presence]/sum(pi_2)
        pi[(ell-1)*N+1:ell*N] = pi_transient

        M_transient = M[(ell-1)*N+1:ell*N,:]
        M_transient[presence,:] = M_transient[presence,:]./pi_2[presence]
        M[(ell-1)*N+1:ell*N,:] = M_transient
    end

    xi = spzeros(Float64,L*N)
    for ell = 1:L
        xi_transient = spzeros(Int64,N)
        presence = findall(isequal(ell),xi_seq[:,3])
        xi_transient[xi_seq[presence,1]] = xi_seq[presence,2]
        xi[(ell-1)*N+1:ell*N] = xi_transient
    end

    # obtain beta_i
    presence_list = findall(!iszero,pi[1:N])
    N1 = length(presence_list)
    absence_list = findall(iszero,pi[1:N])
    MatA_i = M[1:N,:]-sparse(Matrix(1.0I, N, N))
    pi_list = findall(!isequal(presence_list[1]),[i for i = 1:N])
    MatA_i[:,pi_list] = MatA_i[:,pi_list] - MatA_i[:,presence_list[1]]*transpose(pi[pi_list]/pi[presence_list[1]])
    MatA_i_reduced = MatA_i[pi_list,pi_list]

    xi_hat = sum(pi[1:N].*xi[1:N])
    MatB_i = (xi_hat*ones(Float64,N)-xi[1:N])*N1
    MatB_i_reduced = MatB_i[pi_list]

    beta_i_solution_reduced = idrs(MatA_i_reduced, MatB_i_reduced)
    beta_i_solution = zeros(Float64, N)
    beta_i_solution[pi_list] = beta_i_solution_reduced
    beta_i_solution[presence_list[1]] = -sum(pi[1:N].*beta_i_solution)/pi[presence_list[1]]

    # obtain beta_ij
    Mat1 = M[1:N,:]
    inds = findall(!iszero,Mat1)
    a = getindex.(inds, 1)
    b = getindex.(inds, 2)
    vec1 = [i for i = 1:N]
    vec1 = transpose(vec1)
    X1 = N*(a.-1)*ones(Int64,1,N)+ones(Int64,length(a))*vec1
    Y1 = N*(b.-1)*ones(Int64,1,N)+ones(Int64,length(b))*vec1
    W1 = Mat1[inds]*ones(Int64,1,N)

    vec2 = [(i-1)*N for i = 1:N]
    vec2 = transpose(vec2)
    X2 = a*ones(Int64,1,N)+ones(Int64,length(a))*vec2
    Y2 = b*ones(Int64,1,N)+ones(Int64,length(b))*vec2
    W2 = Mat1[inds]*ones(Int64,1,N)

    X11 = reshape(X1,:,1)
    Y11 = reshape(Y1,:,1)
    W11 = reshape(W1,:,1)
    X22 = reshape(X2,:,1)
    Y22 = reshape(Y2,:,1)
    W22 = reshape(W2,:,1)
    MatA_ij = sparse(X11[:,1],Y11[:,1],W11[:,1],N^2,N^2)/2+sparse(X22[:,1],Y22[:,1],W22[:,1],N^2,N^2)/2
    MatA_ij = MatA_ij - sparse(Matrix(1.0I, N*N, N*N))
    beta_obtained = [(i-1)*N+i for i = 1:N]
    beta_missed = setdiff([i for i = 1:N^2], beta_obtained)
    MatA_ij_reduced = MatA_ij[beta_missed, beta_missed]

    MatB_ij = -sum(MatA_ij[:,beta_obtained].*transpose(beta_i_solution), dims = 2)
    Mat = xi_hat*ones(Float64,N,N)*N1/2-xi[1:N]*transpose(xi[1:N])*N1/2
    Mat[absence_list,:] = spzeros(Float64,length(absence_list),N)
    Mat[:,absence_list] = spzeros(Float64,N,length(absence_list))
    MatB_ij = MatB_ij+reshape(transpose(Mat),:)
    MatB_ij_reduced = MatB_ij[beta_missed]

    beta_ij_solution_reduced = idrs(MatA_ij_reduced, MatB_ij_reduced)
    beta_ij_solution = zeros(Float64, N*N)
    beta_ij_solution[beta_missed] = beta_ij_solution_reduced
    beta_ij_solution[beta_obtained] = beta_i_solution

    solution = spzeros(Float64, N*N, L)
    solution[:,1] = beta_ij_solution
    # obtain gamma_ij
    if L > 1
        for ell = 2:L
            xi_ell = xi[(ell-1)*N+1:ell*N]
            pi_ell = pi[(ell-1)*N+1:ell*N]
            presence_ell = findall(!iszero, pi_ell)
            N_ell = length(presence_ell)
            GammaA_ij = sparse(X11[:,1],Y11[:,1],W11[:,1],N^2,N^2)*(N_ell-1)/(N1+N_ell-1)

            Mat_ell = M[(ell-1)*N+1:ell*N,:]
            inds_ell = findall(!iszero,Mat_ell)
            a_ell = getindex.(inds_ell, 1)
            b_ell = getindex.(inds_ell, 2)
            vec_ell = [(i-1)*N for i = 1:N]
            vec_ell = transpose(vec_ell)
            X_ell = a_ell*ones(Int64,1,N)+ones(Int64,length(a_ell))*vec_ell
            Y_ell = b_ell*ones(Int64,1,N)+ones(Int64,length(b_ell))*vec_ell
            W_ell = Mat_ell[inds_ell]*ones(Int64,1,N)
            X_ellell = reshape(X_ell,:,1)
            Y_ellell = reshape(Y_ell,:,1)
            W_ellell = reshape(W_ell,:,1)
            GammaA_ij = GammaA_ij+sparse(X_ellell[:,1],Y_ellell[:,1],W_ellell[:,1],N^2,N^2)*(N1-1)/(N1+N_ell-1)

            vec_gamma1 = reshape(ones(Int64, N)*vec1,:)
            vec_gamma2 = reshape(transpose(vec1)*ones(Int64,1,N),:)
            X_gamma1 = N*(a.-1)*ones(Int64,1,N*N)+ones(Int64,length(a))*transpose(vec_gamma1)
            Y_gamma1 = N*(b.-1)*ones(Int64,1,N*N)+ones(Int64,length(b))*transpose(vec_gamma2)
            W_gamma1 = Mat1[inds]*ones(Int64,1,N*N)

            vec_gamma3 = [(i-1)*N for i in vec_gamma1]
            vec_gamma4 = [(i-1)*N for i in vec_gamma2]
            X_gamma2 = a_ell*ones(Int64,1,N*N)+ones(Int64,length(a_ell))*transpose(vec_gamma3)
            Y_gamma2 = b_ell*ones(Int64,1,N*N)+ones(Int64,length(b_ell))*transpose(vec_gamma4)
            W_gamma2 = Mat_ell[inds_ell]*ones(Int64,1,N*N)

            X_gamma_11 = reshape(X_gamma1,:,1)
            Y_gamma_11 = reshape(Y_gamma1,:,1)
            W_gamma_11 = reshape(W_gamma1,:,1)
            X_gamma_22 = reshape(X_gamma2,:,1)
            Y_gamma_22 = reshape(Y_gamma2,:,1)
            W_gamma_22 = reshape(W_gamma2,:,1)
            GammaA_ij = GammaA_ij + sparse(X_gamma_11[:,1],Y_gamma_11[:,1],W_gamma_11[:,1],N^2,N^2).*sparse(X_gamma_22[:,1],Y_gamma_22[:,1],W_gamma_22[:,1],N^2,N^2)/(N1+N_ell-1)

            GammaA_ij = GammaA_ij - sparse(Matrix(1.0I, N*N, N*N))
            pi_transient = pi[1:N].*pi_ell
            presence_list = findall(!iszero,pi_transient)
            absence_list = findall(iszero,pi_transient)
            GammaA_delete = (presence_list.-1)*N+presence_list
            GammaA_ij[:,GammaA_delete] = GammaA_ij[:,GammaA_delete] - GammaA_ij[:,GammaA_delete[1]]*transpose(pi[presence_list])/pi[presence_list[1]]
            pi_list = findall(!isequal(GammaA_delete[1]),[i for i = 1:N*N])
            GammaA_ij_reduced = GammaA_ij[pi_list,pi_list]

            xi_ell_hat = sum(pi_ell.*xi_ell)
            GammaB_ij = (xi_hat*xi_ell_hat*ones(Float64,N,N) - xi[1:N]*transpose(xi_ell))*N1*N_ell/(N1+N_ell-1)
            presence1 = findall(iszero, pi[1:N])
            presence1 = presence1[:,1]
            presence_ell = findall(iszero, pi_ell)
            presence_ell = presence_ell[:,1]
            GammaB_ij[presence1,:] = spzeros(Float64, length(presence1), N)
            GammaB_ij[:,presence_ell] = spzeros(Float64, N, length(presence_ell))
            GammaB_ij  = reshape(transpose(GammaB_ij), :)
            GammaB_ij_reduced = GammaB_ij[pi_list]

            gamma_ij_solution_reduced = idrs(GammaA_ij_reduced, GammaB_ij_reduced)
            gamma_ij_solution = zeros(Float64, N*N)
            gamma_ij_solution[pi_list] = gamma_ij_solution_reduced
            gamma_ij_solution[GammaA_delete[1]] = -sum(pi[presence_list].*gamma_ij_solution[GammaA_delete])/pi[presence_list[1]]
            solution[:,ell] = gamma_ij_solution
        end
    end

    return solution
end

# provide the value of coefficient of Eq. (2), namely theta and phi
# input---
# multilayer network in the form of an edge sequanece: edge_seq [node id, node id, edge weight, layer id]
# initial strategy configuration: xi_seq [node id, strategy, layer id]
# for an edge between node i and j in layer ll with weight wij_ll, two terms should be added into the edge sequence, i.e. [i j wij_ll ll] and [j i wij_ll ll]
# the code here requires wij_ll to be a positive integer
# output---
# strategy assortment:  theta_phi[:,1] = [theta0, theta1, theta2, theta3], strategy correlation within layer 1
#                       theta_phi[:,ll] = [phi00, phi01, phi20, phi21], strategy correlation between layer 1 and layer ll
# if there are only two layers, in Eq. (2) in the main text, [theta0, theta1, theta2, theta3] = theta_phi[:,1] and [phi00, phi01, phi20, phi21] = theta_phi[:,2]
function bc_multilayer_DB(edge_seq::Array{Int64,2}, xi_seq::Array{Int64,2})
    edge_seq = edge_seq_rearrangement(edge_seq)
    solution = beta_gamma(edge_seq, xi_seq)
    N = maximum(edge_seq[:,1])
    L = maximum(edge_seq[:,4])
    M = spzeros(Float64,N*L,N)
    for ell = 1:L
        edge_list_ell = findall(isequal(ell),edge_seq[:,4])
        M_ell = sparse(edge_seq[edge_list_ell,1],edge_seq[edge_list_ell,2],edge_seq[edge_list_ell,3],N,N)
        M[(ell-1)*N+1:ell*N,:] = M_ell
    end
    pi = spzeros(Float64,L*N)
    for ell = 1:L
        pi_2 = sum(M[(ell-1)*N+1:ell*N,:], dims = 2)
        pi_2 = pi_2[:,1]
        pi_transient = spzeros(Float64, N)
        presence = findall(!iszero,pi_2)
        pi_transient[presence] = pi_2[presence]/sum(pi_2)
        pi[(ell-1)*N+1:ell*N] = pi_transient

        M_transient = M[(ell-1)*N+1:ell*N,:]
        M_transient[presence,:] = M_transient[presence,:]./pi_2[presence]
        M[(ell-1)*N+1:ell*N,:] = M_transient
    end

    M1 = M[1:N,:]
    beta = transpose(reshape(solution[:,1], N, :))
    theta1 = sum(pi[1:N].*M1.*beta)
    M2 = M1*M1
    theta2 = sum(pi[1:N].*M2.*beta)
    M3 = M2*M1
    theta3 = sum(pi[1:N].*M3.*beta)

    theta_phi = zeros(Float64, 4, L)
    theta_phi[:,1] = [0.0,theta1,theta2,theta3]

    if L > 1
        for ell = 2:L
            gamma = transpose(reshape(solution[:,ell], N, :))
            M_ell = M[(ell-1)*N+1:ell*N,:]
            phi01 = sum(pi[1:N].*M_ell.*gamma)
            phi20 = sum(pi[1:N].*M2.*gamma)
            M_ell21 = M2*M_ell
            phi21 = sum(pi[1:N].*M_ell21.*gamma)
            theta_phi[:,ell] = [0,phi01,phi20,phi21]
        end
    end

    return theta_phi
end


# an example about the using of above functions
# see Fig. 5A in the main text, where nodes are labelled and initial strategy configuration is presented in SFig. 2
# the edge sequence is given in the form [node id, node id, edge weight, layer id]
edge_seq = [1 6 1 1;
            2 6 1 1;
            3 6 1 1;
            4 6 1 1;
            5 6 1 1;
            6 1 1 1;
            6 2 1 1;
            6 3 1 1;
            6 4 1 1;
            6 5 1 1;
            1 2 1 2;
            1 3 1 2;
            1 4 1 2;
            1 5 1 2;
            1 6 1 2;
            2 1 1 2;
            3 1 1 2;
            4 1 1 2;
            5 1 1 2;
            6 1 1 2]
# the strategy configuration is given in the form [node id, strategy, layer id]
xi_seq = [1 1 1;
          2 0 1;
          3 0 1;
          4 0 1;
          5 0 1;
          6 0 1;
          1 0 2;
          2 0 2;
          3 0 2;
          4 0 2;
          5 0 2;
          6 1 2]
# see Eq.(2) in the main text
# theta_phi[:,1] = [theta0, theta1, theta2, theta3], strategy correlation within layer 1
# theta_phi[:,2] = [phi00, phi01, phi20, phi21], strategy correlation between layer 1 and layer 2
theta_phi = bc_multilayer_DB(edge_seq, xi_seq)
