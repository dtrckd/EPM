%%Demo code for Infinite Edge Partition Models (EPMs)

clear;

% dataset = 'Toy';
% dataset = 'protein230';
% dataset = 'NIPS12';
% dataset = 'NIPS234';
% dataset = 'football';

%state = 0,1,2,3,4\\
%state = 0;

%datasets = {'manufacturing'};
%datasets = {'link-dynamic-simplewiki'};
datasets = {'manufacturing', 'moreno_names', 'fb_uc'};

outp = '/home/dtrckd/Desktop/workInProgress/networkofgraphs/process/repo/ml/data/mat/';
n_workers = 2;


training_ratio = '100';
testset_ratio = '20';
validset_ratio = '10'; % put it in the training set

K=10;

ratio_id = ['_',training_ratio,'-',testset_ratio,'-',validset_ratio];


p = parpool('local', n_workers)
f(1:length(datasets)) = parallel.FevalFuture;

%%  Run the models
for idx=1:length(datasets)

    dataset = datasets{idx}

    %%% Create directory
    if ~exist('results/')
        mkdir('results')
    end
    if ~exist(['results/', dataset])
        mkdir(['results/', dataset])
    end

    %%% Expe Setup

    %Burnin=500;
    %Collections=500;
    Burnin=5;
    Collections=10;

    Datatype='Count';
    %Datatype='Binary';
    %Modeltype = 'Infinite';
    Modeltype = 'Finite';
    IsDisplay = false;

    %%% Load dataset
    Data = load(strcat(outp, dataset, ratio_id, '.mat'));
    B = Data.Y;
    Ytest = Data.Ytest;
    state = Data.state;
    is_symmetric = Data.is_symmetric;
    if is_symmetric
        B = triu(B,1);
    end

    N = size(B,2);

    %% FIX the validset as faire comparison to wwwsb
    vld = 2 * str2double(validset_ratio) / 100;
    n_valid = round(N*vld / (1+vld));
    idx_test = find(Ytest);
    idx_valid = datasample(idx_test, n_valid, 'replace', false);
    Ytest(idx_valid) = 0;


    %% HGP_EPM: Hierachical gamma process edge partition model
    % rand('state',state);
    % randn('state',state);
    rng(state,'twister');

    %[AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z,Wreal, Wpred, Wpred2, WSIM, WSIM2] = HGP_EPM2(B,K, nan,Ytest,Burnin, Collections, IsDisplay, Datatype, Modeltype, is_symmetric);
    f(idx) = parfeval(p, @HGP_EPM2, 17, B, K, nan, Ytest, Burnin, Collections, IsDisplay, Datatype, Modeltype, is_symmetric);

end

for idx_=1:length(datasets)

    [idx,timing,AUC,AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z,Wreal, Wpred, Wpred2, WSIM, WSIM2] = fetchNext(f);
    dataset = datasets{idx};
    fprintf('HGP_EPM %s, AUCroc = %.2f, WSIM = %.2f, WSIM2 = %.2f, Time = %.0f seconds\n', dataset, AUCroc, WSIM, WSIM2, timing);
    f(idx_).Diary

    it_ = int2str(Burnin+Collections);
    save(['results/', dataset, '/', 'wsim_all',it_,ratio_id,'.mat'], 'Wreal', 'Wpred', 'Wpred2', 'WSIM', 'WSIM2', 'AUCroc', 'timing', 'AUC')
end





