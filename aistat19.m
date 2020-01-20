%%Demo code for Infinite Edge Partition Models (EPMs)

clear;

% dataset = 'Toy';
% dataset = 'protein230';
% dataset = 'NIPS12';
% dataset = 'NIPS234';
% dataset = 'football';

%state = 0,1,2,3,4\\
%state = 0;
%
outp = '/home/dtrckd/Desktop/workInProgress/networkofgraphs/process/repo/ml/data/mat/';

%datasets = {'manufacturing'};
%datasets = {'link-dynamic-simplewiki'};
%datasets = {'fb_uc', 'moreno_names', 'manufacturing',};
%datasets = {'fb_uc', 'moreno_names', 'manufacturing',};
dataset = {
	'fb_uc',
	'hep-th',
	'link-dynamic-simplewiki',
	'enron',
	'slashdot-threads',
	'prosper-loans',
	'munmun_digg_reply',
	'moreno_names',
	'astro-ph'
	};

testset_ratio = '20';
validset_ratio = '10'; % put it in the training set

%training_ratios = {'100', '20'};
training_ratios = {'20', '100'};
repeats = {'1', '2', '3', '4'};
Ks={10,20,30,50};

n_workers = 2;

p = parpool('local', n_workers)
f(1:length(datasets)) = parallel.FevalFuture;

n_expe = length(datasets)*length(training_ratios)*length(repeats)*length(Ks);

%%  Run the models
idx = 0;
for dataset_=1:length(datasets)
for training_ratio_=1:length(training_ratios)
for K_=1:length(Ks)
for repeat_=1:length(repeats)

    idx = idx+1;
    dataset = datasets{dataset_};
    training_ratio = training_ratios{training_ratio_};
    K = Ks{K_};
    repeat = repeats{repeat_};

    expe_state = {};
    expe_state.dataset = dataset;
    expe_state.training_ratio = training_ratio;
    expe_state.K = K;
    expe_state.repeat = repeat;


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
    Burnin=200;
    Collections=100;

    Datatype='Count';
    %Datatype='Binary';
    %Modeltype = 'Infinite';
    Modeltype = 'Finite';
    IsDisplay = false;

    %%% Load dataset
    if isnan(repeat)
        ratio_id = ['_',training_ratio,'-',testset_ratio,'-',validset_ratio];
        fnin = strcat(outp, dataset, ratio_id, '.mat');
    else
        ratio_id = ['_',training_ratio,'-',testset_ratio,'-',validset_ratio];
        fnin = strcat(outp, repeat,'/', dataset, ratio_id, '.mat');
    end
    fprintf('%s - reading in: %s\n', dataset, fnin);
    Data = load(fnin);
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
    f(idx) = parfeval(p, @HGP_EPM2, 18, expe_state, B, K, nan, Ytest, Burnin, Collections, IsDisplay, Datatype, Modeltype, is_symmetric);

end
end
end
end


for idx_=1:n_expe

    [idx,expe_state,timing,AUC,AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z,Wreal, Wpred, Wpred2, WSIM, WSIM2] = fetchNext(f);

    dataset = expe_state.dataset;
    repeat = expe_state.repeat;

    fprintf('HGP_EPM %s, AUCroc = %.2f, WSIM = %.2f, WSIM2 = %.2f, Time = %.0f seconds\n', dataset, AUCroc, WSIM, WSIM2, timing);
    f(idx).Diary

    format_id = ['it',int2str(Burnin+Collections),'training',training_ratio,'K',int2str(K),'rep',repeat];

    fnout = ['results/', dataset, '/', 'wsim_all_',format_id,ratio_id,'.mat'];
    fprintf('writing in: %s\n', fnout);
    save(fnout, 'Wreal', 'Wpred', 'Wpred2', 'WSIM', 'WSIM2', 'AUCroc', 'timing', 'AUC')
end





