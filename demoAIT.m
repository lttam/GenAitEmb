clear all;
clc;

% setup parameters for the dataset
datasetNAME = 'MIT8'; % dim = 100
typeDATA = 'ORG';
kMAX = 20; % k in kNN classifier

dirFolderIn = 'Data/'; % path for data
dirFolderOut = 'Data/'; % path to save results
mkdir(dirFolderOut);

fileMAT = [dirFolderIn datasetNAME '_' typeDATA '.mat']; % load dataset
load(fileMAT);

numClass = length(unique(yTr)); % number of class

tic;
[L, b] = AIT_PSGD_NES(xTr, yTr); % main algorithm
runTime = toc;

XX = [1:kMAX];          % for plot (X-Axis)
YY = zeros(size(XX));   % Accuracy for each k in kNN (average over samples)
yCLASS = zeros(kMAX, numClass); % predict for each samples with each k
yAvgCLASS = zeros(size(XX));    % Accuracy (average over class)

indexFILE = [dirFolderIn datasetNAME '_testIndex.mat']; % starting & ending index for each class in test set
% sID: starting index, eID: ending index
load(indexFILE);

n = length(yTe);
predKMAX = KNN_AIT(yTr, xTr, L, kMAX, xTe, b);  % kNN for Generalized Aitchison Embeddings 
   
for kk = 1:kMAX
    pred = predKMAX(kk, :);
    acc = sum(pred==yTe)/n;
    YY(kk) = acc;
    for ii=1:numClass
        nSample = eID(ii) - sID(ii) + 1;
        yCLASS(kk, ii) = sum( pred(sID(ii):eID(ii)) == yTe(sID(ii):eID(ii)) )/nSample;
    end
    yAvgCLASS(kk) = sum(yCLASS(kk, :))/numClass;
end

save([dirFolderOut datasetNAME '_AITCHISON.mat'], 'XX', 'YY', 'yCLASS', 'yAvgCLASS', 'L', 'b', 'runTime');

disp('FINISH !!!');
