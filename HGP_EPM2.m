function [AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z, Wreal, Wpred, WSIM, WSIM2]=HGP_EPM2(B, K, idx_train,Ytest,Burnin, Collections, IsDisplay, Datatype, Modeltype, is_symmetric)
%Code for Hierachical Gamma Process Edge Partition Model
%Mingyuan Zhou, Oct, 2014
%Input:
%B is an upper triagular matrix, the diagonal terms are not defined
%idx_train: training indices
%Ytest: sparse matrice of test indices
%K: truncated number of atoms
%Datatype: 'Count' or 'Binary'. Use 'Count' for integer-weigted edges.


%Output:
%Phi: each row is a node's feature vector
%r_k: each elment indicates a community's popularity
%Lambda_KK: community-community interaction strengh rate matrix
%AUCroc: area under the ROC curve
%AUCpr: area under the precition-recall curve
%m_i_k_dot_dot: m_i_k_dot_dot(k,i) is the count that node $i$ sends to community $k$
%ProbeAve: ProbeAve(i,j) is the estimated probability for nodes $i$ and $j$ to be linked
%z: hard community assignment

if ~exist('K','var')
    K = 100;
end
if ~exist('Burnin','var')
    Burnin = 2000;
end
if ~exist('Collections','var')
    Collections = 500;
end
if ~exist('Datatype','var')
    Datatype = 'Binary';
    %Datatype = 'Count';
end

if ~exist('Modeltype','var')
    Modeltype = 'Infinite';
    %Modeltype = 'Finite';
end

iterMax = Burnin+Collections;
N = size(B,2);

idx_test = find(Ytest);

%BTrain_Mask = sparse(zeros(size(B)));
%BTrain_Mask(~idx_test) = 1;
%BTrain_Mask=BTrain_Mask+BTrain_Mask';


BTrain = B;
BTrain(idx_test) = 0;

[ii,jj,M]=find(BTrain);
idx  = sub2ind([N,N],ii,jj);


Phi = gamrnd(1e-0*ones(N,K),1);


output.K_positive = zeros(1,iterMax);
output.K_hardassignment= zeros(1,iterMax);
output.Loglike_Train = zeros(1,iterMax);
output.Loglike_Test = zeros(1,iterMax);
AUC = zeros(1,iterMax);

ProbSamples = zeros(N,N);
r_k=ones(K,1)/K;

Wpred_samples = zeros(size(idx_test));
Wsim2_samples = 0;



%Parameter Initialization
Epsilon = 1;
beta1=1;
beta2=1;

Lambda_KK=r_k*r_k';
if is_symmetric
    Lambda_KK = triu(Lambda_KK,1)+triu(Lambda_KK,1)';
end
Lambda_KK(sparse(1:K,1:K,true))=Epsilon*r_k;
gamma0=1;
c0=1;
a_i=0.01*ones(N,1);
e_0=1e-0;
f_0=1e-0;
c_i = ones(N,1);
count=0;
IsMexOK=true;
LogLikeMax = -inf;
Kmin=inf;
EPS=0.01;
EPS=0;


for iter=1:iterMax
    
    %draw a latent count for each edge
    if strcmp(Datatype, 'Binary')
        Rate = sum((Phi(ii,:)*Lambda_KK).*Phi(jj,:),2);
        M = truncated_Poisson_rnd(Rate);
    end
    
    %Sample m_i_k1_k2_j and update m_i_k_dot_dot and m_dot_k_k_dot
    if IsMexOK
        [m_i_k_dot_dot, m_dot_k_k_dot] = Multrnd_mik1k2j(sparse(ii,jj,M,N,N),Phi,Lambda_KK);
    else
        m_i_k_dot_dot = zeros(K,N);
        m_dot_k_k_dot = zeros(K,K);
        for ij=1:length(idx)
            pmf = (Phi(ii(ij),:)'*Phi(jj(ij),:)).*Lambda_KK;
            mij_kk = reshape(multrnd_histc(M(ij),pmf(:)),K,K);
            m_i_k_dot_dot(:,ii(ij)) = m_i_k_dot_dot(:,ii(ij)) + sum(mij_kk,2);
            m_i_k_dot_dot(:,jj(ij)) = m_i_k_dot_dot(:,jj(ij)) + sum(mij_kk,1)';
            m_dot_k_k_dot = m_dot_k_k_dot + mij_kk + mij_kk';
        end
    end
    m_dot_k_k_dot(sparse(1:K,1:K,true))=m_dot_k_k_dot(sparse(1:K,1:K,true))/2;
    
    %Number of communities assigned with nonzero counts
    output.K_positive(iter) = nnz(sum(m_i_k_dot_dot,2));
    
        
    %Sample a_i and phi_ik
    Phi_times_Lambda_KK = Phi*Lambda_KK;

    for i=randperm(N)
        ell = CRT_sum_mex(m_i_k_dot_dot(:,i),a_i(i));
        p_ik_prime_one_minus = c_i(i)./(c_i(i)+(~Ytest(i,:))*Phi_times_Lambda_KK);
        a_i(i) = gamrnd(ell+1e-2,1./(1e-2-sum(log(max(p_ik_prime_one_minus,realmin)))));
        Phi(i,:) =  randg(a_i(i) + m_i_k_dot_dot(:,i))'.*p_ik_prime_one_minus/c_i(i);
        Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
    end

    %Sample c_i
    c_i = gamrnd(1e-0 + K*a_i,1./(1e-0 +  sum(Phi,2)));
    
    %Phi_KK(k_1,k_2) = 2^{-\delta(k_2=k_1)} \sum_{i}\sum_{j\neq i} \phi_{ik_1} \phi_{jk_2}

    %Phi_KK = Phi'*BTrain_Mask*Phi;
    Phi_KK = (repmat(sum(Phi', 2), 1, N) - Phi'*Ytest)*Phi;

    Phi_KK(sparse(1:K,1:K,true)) = Phi_KK(sparse(1:K,1:K,true))/2;
    
    triu1dex = triu(true(K),1);
    triu2dex = (true(K) - eye(K)) > 0 ;
    diagdex = sparse(1:K,1:K,true);
    
    %Sample r_k
    L_KK=zeros(K,K);
    temp_p_tilde_k=zeros(K,1);
    p_kk_prime_one_minus = zeros(K,K);
    for k=randperm(K)
        R_KK=r_k';
        R_KK(k)=Epsilon;
        beta3=beta2*ones(1,K);
        beta3(k)=beta1;
        p_kk_prime_one_minus(k,:) = beta3./(beta3+ Phi_KK(k,:));
        
        if strcmp(Modeltype, 'Infinite')
            L_KK(k,:) = CRT_sum_mex_matrix(sparse(m_dot_k_k_dot(k,:)),r_k(k)*R_KK);
            temp_p_tilde_k(k) = -sum(R_KK.*log(max(p_kk_prime_one_minus(k,:), realmin)));
            r_k(k) = randg(gamma0/K+sum(L_KK(k,:)))./(c0+temp_p_tilde_k(k));
        end
    end
    
    if strcmp(Modeltype, 'Infinite')
        %Sample gamma0 with independence chain M-H
        ell_tilde = CRT_sum_mex(sum(L_KK,2),gamma0/K);
        sum_p_tilde_k_one_minus = -sum(log(c0./(c0+temp_p_tilde_k) ));
        gamma0new = randg(e_0 + ell_tilde)./(f_0 + 1/K*sum_p_tilde_k_one_minus);
        %AcceptProb1 = exp(sum(gampdfln(max(r_k,realmin),gamma0new/K,c0))+gampdfln(gamma0new,e_0,f_0) + gampdfln(gamma0,e_0 + ell_tilde,f_0-1/K* sum(log(c0./(c0+temp_p_tilde_k) ))  )...
        %         -(sum(sum(gampdfln(max(r_k,realmin),gamma0/K,c0))+gampdfln(gamma0,e_0,f_0) + gampdfln(gamma0new,e_0 + ell_tilde,f_0-1/K* sum(log(c0./(c0+temp_p_tilde_k) )) ) )   ));
        AcceptProb1 = CalAcceptProb1(r_k,c0,gamma0,gamma0new,ell_tilde,1/K*sum_p_tilde_k_one_minus,K);
        if AcceptProb1>rand(1)
            gamma0=gamma0new;
            count =count+1;
            %count/iter
        end
        %gamma0 =0.01;
        %Sample c0
        c0 = randg(1 + gamma0)/(1+sum(r_k));
    end
    % gamma0 = 0.1;
    %Sample Epsilon
    ell = sum(CRT_sum_mex_matrix( sparse(m_dot_k_k_dot(diagdex))',Epsilon*r_k'));
    %if iter>100
    Epsilon = randg(ell+1e-2)/(1e-2-sum(r_k.*log(max(p_kk_prime_one_minus(diagdex), realmin))));
    % Epsilon=1;
    %end
    
    %Sample lambda_{k_1 k_2}
    R_KK = r_k*(r_k');
    R_KK(sparse(1:K,1:K,true)) = Epsilon*r_k;
    Lambda_KK=zeros(K,K);
    Lambda_KK(diagdex) = randg(m_dot_k_k_dot(diagdex) + R_KK(diagdex))./(beta1+Phi_KK(diagdex));
    if is_symmetric
        Lambda_KK(triu1dex) = randg(m_dot_k_k_dot(triu1dex) + R_KK(triu1dex))./(beta2+Phi_KK(triu1dex));
        Lambda_KK = Lambda_KK + triu(Lambda_KK,1)'; %Lambda_KK is symmetric
    else
        Lambda_KK(triu2dex) = randg(m_dot_k_k_dot(triu2dex) + R_KK(triu2dex))./(beta2+Phi_KK(triu2dex));
    end
    
    %     if iter<100
    %         Lambda_KK=Lambda_KK-triu(Lambda_KK,2);
    %         Lambda_KK = triu(Lambda_KK,1)+triu(Lambda_KK,1)';
    %     end
    
    %Sample beta1 and beta2
    %     beta1 = randg(sum(R_KK(diagdex))+1e-2)./(1e-2+ sum(Lambda_KK(diagdex)));
    %     beta2 = randg(sum(R_KK(triu1dex))+1e-2)./(1e-2+ sum(Lambda_KK(triu1dex)));
    %
    if is_symmetric
        beta1 = randg(sum(R_KK(diagdex))+sum(R_KK(triu1dex))+1e-0)./(1e-0+ sum(Lambda_KK(diagdex))+sum(Lambda_KK(triu1dex)));
    else
        beta1 = randg(sum(R_KK(diagdex))+sum(R_KK(triu2dex))+1e-0)./(1e-0+ sum(Lambda_KK(diagdex))+sum(Lambda_KK(triu2dex)));
    end
    beta2 = beta1;
    
    Prob = Phi*(Lambda_KK)*Phi'+EPS;
    aWsim2 = wsim2(B, idx_test, Phi, Lambda_KK, is_symmetric);
    
    %% WSIM
    aWpred = Prob(idx_test);

    %fprintf('%.2f %.2f\n', aWpred, aWsim2)
    
    Prob = 1-exp(-Prob);
    if iter>Burnin
        %ProbSamples(:,:,iter-Burnin) = Prob;
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples/(iter-Burnin);

        Wpred_samples = Wpred_samples + aWpred;
        Wpred = Wpred_samples / (iter-Burnin);

        Wsim2_samples = Wsim2_samples + aWsim2;
        WSIM2 = Wsim2_samples / (iter-Burnin);
    else
        ProbAve = Prob;
        Wpred = aWpred;
        WSIM2 = aWsim2;
    end
    
    
    %  rate = 1-exp(-ProbAve(idx_test));
    rate= Prob(idx_test);
    links = full(double(B(idx_test)>0));
    AUC(iter) = aucROC(rate,links);
    %[~,~,~,AUC(iter)] = perfcurve(links,rate,1);
    
    
    
    
    z=zeros(1,N);
    
    %if iter<Burnin
    % [~,rdex]=sort(sum(mi_dot_k,2));
    %  [~,rdex]=sort(sum(m_i_k_dot_dot,2));
    %   [~,rdex]=sort(r_k);
    [~,rdex]=sort(-sum(m_dot_k_k_dot,2));
    %rdex=1:K;
    %Phir=Phi*sqrt(diag(r));
    
    
    rrr = diag(Lambda_KK);
    
    
    
    
    
    %     n_k = sparse(z,1,1,K,1);
    %     [~,zdex]=sort(n_k);
    %     kkkk=1:K;
    %     kkkk=kkkk(zdex);
    %     yy=zeros(1,K);
    %     yy(kkkk)=1:K;
    %     z=yy(z);
    
    %n_k = sparse(z,1,1);
    [~,rdex]=sort(sum(m_i_k_dot_dot,2),'descend');
    [~,z] = max(m_i_k_dot_dot(rdex,:),[],1);
    [~,Rankdex]=sort(z);
    output.K_hardassignment(iter) = length(unique(z));
    
    %  end
    %  Rankdex = 1:N;
    
    %%if output.Loglike_Train(iter)>LogLikeMax && iter>1000
    %% LogLikeMax = output.Loglike_Train(iter);
    %     if output.K_positive(iter)<=Kmin && iter>1000
    %
    %         Kmin = output.K_positive(iter);
    %               Network_plot;
    %
    %     end
    
    if mod(iter,1000)==0
        fprintf('Iter= %d, Number of Communities = %d \n',iter, output.K_positive(iter));
    end
end

rate = ProbAve(idx_test);


links = full(double(B(idx_test)>0));
[X,Y,T,AUCroc] = perfcurve(links,rate,1);
[prec, tpr, fpr, thresh] = prec_rec(rate, links,  'numThresh',3000);
AUCpr = trapz([0;tpr],[1;prec]);
F1= max(2*tpr.*prec./(tpr+prec));


Wreal = full(B(idx_test));
WSIM = mean((Wreal - Wpred).^2); % MSE
WSIM2 = WSIM2;
