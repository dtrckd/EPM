%%Demo code for Infinite Edge Partition Models (EPMs)

clear;

% dataset = 'Toy';
% dataset = 'protein230';
% dataset = 'NIPS12';
% dataset = 'NIPS234';
% dataset = 'football';

%state = 0,1,2,3,4\\

datasets = {'manufacturing'};
datasets = {'manufacturing', 'moreno_names'};
outp = '/home/dtrckd/Desktop/workInProgress/networkofgraphs/process/repo/ml/data/mat/';
state = 0;

Burnin=1500;
Collections=1500;

TrainRatio =.8;



%%  Run the models

Datatype='Count';
%Datatype='Binary';
%Modeltype = 'Infinite';
Modeltype = 'Finite';
IsDisplay = false;

for dataset_=datasets

    dataset=dataset_{:}

    %if strcmp(dataset, 'manufacturing')
        Data = load(strcat(outp, dataset, '.mat'));
        B=Data.Y;
        Ytest=Data.Ytest;
        state = Data.state;
        is_symmetric = Data.is_symmetric;
        if is_symmetric
            B = triu(B,1);
        end

        N = size(B,2);
        K=10;
    %end


    %% HGP_EPM: Hierachical gamma process edge partition model
    % rand('state',state);
    % randn('state',state);
    rng(state,'twister');

    tic
    [AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z,Wreal, Wpred,WSIM, WSIM2]=HGP_EPM2(B,K, nan,Ytest,Burnin, Collections, IsDisplay, Datatype, Modeltype, is_symmetric);
    fprintf('HGP_EPM, AUCroc =  %.4f, AUCpr = %.4f, Time = %.0f seconds \n',AUCroc,AUCpr,toc);

    if state==0
        save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr','F1','Phi','Lambda_KK','r_k','ProbAve','m_i_k_dot_dot','output','z');
    else
        save(['results/',dataset,num2str(state),'HGP_EPM.mat'],'AUCroc','AUCpr');
    end
    save(['results/',dataset,'/','wsim_all.mat'], 'Wreal', 'Wpred', 'WSIM', 'WSIM2')


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

end



