%%Demo code for Infinite Edge Partition Models (EPMs)

% dataset = 'Toy';
% dataset = 'protein230';
% dataset = 'NIPS12';
% dataset = 'NIPS234';
% dataset = 'football';

%state = 0,1,2,3,4\\

dataset = 'football';
state = 0;

Burnin=1500;
Collections=1500;

TrainRatio =.8;

%for state = 0:4
if strcmp(dataset,'football')
    Data = load('data/football_corrected.mat');
    B=Data.B;
    N = size(B,2);
    B = triu(B,1);
    K=10;
end


%% Save data to be used by the eigenmodel R package provided by Peter Hoff

%rand('state',state);
%randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
BB=full(B+B');
BBMask=full(BTrain_Mask);
save(['results/',dataset,'_B_',num2str(state),'.mat'], 'BBMask', 'BB');
%%%Run the R code: EigenModel.R
%%
%
%     load(['R',dataset,'_B_',num2str(state),'_',num2str(K),'.mat'])
%     rate = ProbAve(idx_test);
%     figure;
%     subplot(1,2,1)
%     links = double(B(idx_test)>0);
%     [~,dex]=sort(rate,'descend');
%     subplot(2,2,1);plot(rate(dex))
%     subplot(2,2,2);plot(links(dex),'*')
%     subplot(2,2,3);
%     [X,Y,T,AUCroc] = perfcurve(links,rate,1);
%     plot(X,Y);
%     axis([0 1 0 1]), grid on, xlabel('FPR'), ylabel('TPR'), hold on;
%     x = [0:0.1:1];plot(x,x,'b--'), hold off; title(['AUCroc = ', num2str(AUCroc)])
%     subplot(2,2,4)
%     [prec, tpr, fpr, thresh] = prec_rec(rate, links,  'numThresh',3000);
%     plot([0; tpr], [1 ; prec]); % add pseudo point to complete curve
%     xlabel('recall');
%     ylabel('precision');
%     title('precision-recall graph');
%     AUCpr = trapz(tpr,prec);
%     F1= max(2*tpr.*prec./(tpr+prec));
%     title(['AUCpr = ', num2str(AUCpr), '; F1 = ', num2str(F1)])
%     figure;imagesc(normcdf(ProbProbit))
%figure(100);subplot(2,3,4);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(d) AGM')


%%  Run the models

Datatype='Count';
%Datatype='Binary';
%Modeltype = 'Infinite';
Modeltype = 'Finite';
IsDisplay = false;

%% HGP_EPM: Hierachical gamma process edge partition model
% rand('state',state);
% randn('state',state);
rng(state,'twister');
[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
tic
[AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z,Wreal, Wpred,WSIM]=HGP_EPM(B,K, idx_train,idx_test,Burnin, Collections, IsDisplay, Datatype, Modeltype);
fprintf('HGP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

if state==0
    save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr','F1','Phi','Lambda_KK','r_k','ProbAve','m_i_k_dot_dot','output','z');
else
    save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr');
end
save(['results/',dataset,'/','wsim_all.mat'], 'Wreal', 'Wpred', 'WSIM')


%figure(100);subplot(2,3,6);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(f) HGP-EPM')
%figure;imagesc(ProbAve)


%%% GP_EPM: Gamma process edge partition model
%%  rand('state',state);
%%  randn('state',state);
%rng(state,'twister');
%[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
%
%tic
%[AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z] = GP_EPM(B,K,idx_train,idx_test,Burnin,Collections, IsDisplay, Datatype, Modeltype);
%fprintf('GP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);
%
%if state==0
%    save(['results/',dataset,num2str(state),'GP_EPM.mat'],'AUCroc','AUCpr','F1','Phi','r','ProbAve','mi_dot_k','output','z');
%else
%    save(['results/',dataset,num2str(state),'GP_EPM.mat'],'AUCroc','AUCpr');
%end
%figure(100);subplot(2,3,5);imagesc(ProbAve-diag(diag(ProbAve)),[0,1]);title('(e) GP-EPM')
%
%%figure;imagesc(ProbAve)



