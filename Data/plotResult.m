% run demoAIT to create MIT8_AITCHISON.mat file
% run demoLMNN to create MIT8_LMNN.mat file

% This file gives a simple way to plot results of Generalize Aitchison
% Embeddings v.s LMNN (with or with out Hellinger mapping on MIT Scene
% dataset

clear all;
clc;

datasetName = 'MIT8';

% LMNN
load([datasetName '_LMNN.mat']);

tt = YY;
% AITCHISON
load([datasetName '_AITCHISON.mat']);

ff = figure;
hold on
plot(XX, YY, 'r'); % AITCHISON
plot(XX, tt, 'b'); % LMNN
hold off

legend('AITCHISON', 'LMNN');

