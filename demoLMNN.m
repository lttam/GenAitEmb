clear all;
clc;

% flagHELLINGER = 1 --> using Hellinger mapping before learning LMNN (LMNN_HELLINGER)
% flagHELLINGER = 0 --> using original data (without Hellinger mapping before learning LMNN)
flagHELLINGER = 0; % Performance is better when we use Hellinger mapping for data.  

% same format with demoAIT.m
datasetNAME = 'MIT8';
typeDATA = 'ORG';
kMAX = 20;

dirFolderIn = 'Data/';
dirFolderOut = 'Data/';
mkdir(dirFolderOut);

fileMAT = [dirFolderIn datasetNAME '_' typeDATA '.mat'];
load(fileMAT);

numClass = length(unique(yTr)); % number of class

if flagHELLINGER == 1 % using Hellinger mapping for data or not 
    xTr = sqrt(xTr);
    xTe = sqrt(xTe);
end

tic;
L = lmnn2(xTr, yTr);
runTime = toc;

XX = [1:kMAX];
YY = zeros(size(XX));
yCLASS = zeros(kMAX, numClass);
yAvgCLASS = zeros(size(XX)); 

indexFILE = [dirFolderIn datasetNAME '_testIndex.mat'];
% sID, eID
load(indexFILE);

n = length(yTe);
predKMAX = KNN_LMNN(yTr, xTr, L, kMAX, xTe); 
   
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

save([dirFolderOut datasetNAME '_LMNN.mat'], 'XX', 'YY', 'yCLASS', 'yAvgCLASS', 'L', 'runTime');


