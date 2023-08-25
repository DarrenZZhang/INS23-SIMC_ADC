 clear;
 addpath('./data')
 addpath('./measure')
 addpath('./file')
%  clc;

%data:numView*1，each row represents an instance
Dataname='BDGP_fea';
percentDels=[0.1,0.3,0.5];
miu = 1e-2;
rho = 1.2;
load(Dataname);
x=X;
truthF=Y;
clear X Y
num_view=length(x);
% ns=length(unique(truthF));
numFolds=1;
k = length(unique(truthF));
d = k;

for i_percentDel = 1:length(percentDels)
    percentDel = percentDels(i_percentDel);
    Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
    load(Datafold);
    filename = strcat(['SIMC-',Dataname,'-',num2str(percentDel),'.txt']);
    for iter_folds=1:numFolds
        ind_folds = folds{iter_folds}; 

        %% construct incomplete data
        for iv = 1:length(x)
            X1 = x{iv}';
            X1 = NormalizeFea(X1,0);
            ind_0 = find(ind_folds(:,iv) == 0);
            ind_1 = find(ind_folds(:,iv) == 1);
            X1(:,ind_0) = 0;    % construct missing data by replacing them with 0
            Y{iv} = X1;         
            
            W1 = eye(size(X1,2));
            W1(:,ind_1) = [];
            W{iv} = W1';%W:n_v*n
            Ne(iv) = length(ind_0); 
            X_exist{iv}=X1(:,ind_1);
            avg = mean(X_exist{iv},2);
            for ij = 1:length(ind_0)
                E{iv}(:,ij)=avg;
            end
        end
        clear X1 ind_0
        X = Y;
        clear Y
        for iv = 1:num_view
            X{iv} = X{iv}';
        end
        %%
        numanchor=[k,2*k,3*k];
        lambda = 1;
        beta = 1;
        gamma = [1e-1,1,1e1,1e2,1e3,1e4,1e5];

        for j=1:length(numanchor)
            rng(5489,'twister');
            for ibeta = 1:length(beta)
                for i=1:length(lambda)
                    for p=1:length (gamma)
                        tic;
                        [U,~,~,~,~,~,~] = SIMC(X,truthF,lambda,d,numanchor(j),W,Ne,E,beta(ibeta),gamma(p));
                         res = myNMIACCwithmean(U,truthF,k);
                         t=toc;
                         fprintf('time: %f\n',t);
                         fprintf('beta: %f, gamma: %f\n',beta(ibeta),gamma(p));
                        fprintf('Anchor:\t%d \t ACC:%12.6f \t NMI:%12.6f \t Purity:%12.6f \t Fscore:%12.6f \t Time:%12.6f \n',[numanchor(j) res(1) res(2) res(3) res(4) t]);
                        dlmwrite(filename,[numanchor(j) beta(ibeta) lambda(i) gamma(p) res t],'-append','delimiter','\t','newline','pc');
                    end
                end
            end
        end
    end
end
