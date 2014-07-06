function [LBest, bBest] = AIT_PSGD_NES(xTr, yTr, maxIter, stepSize, numUpdate)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 0.1 (May 9th, 2014)
% Projected subgradient descent with Nesterov's acceleration algorithm
%
% Authors: Tam Le @Kyoto University
% (tam.le@iip.ist.i.kyoto-u.ac.jp)
% Advisor: Marco Cuturi @Kyoto University
%
% Relevent papers: 
% [1] Tam Le, Marco Cuturi, Generalized Aitchison Embeddings for Histograms,
% Asian Conference on Machine Learning (ACML), 2013.
% [2] Tam Le, Marco Cuturi, Generalized Aitchison Embeddings for
% Histograms, Machine Learning Journal (MLJ), 2014.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default parameters (basically, it is better to choose them via validation
dfMaxIter = 200; % less than 500 is enough (early stopping is sometimes better), most of cases, 200 or 300 iterations are fine.
dfNumUpdate = 10; % some common # iterations for update {20, 10}
dfStepSize = 0.01; % some common step size {0.1, 0.05, 0.01}

if nargin < 3
    maxIter = dfMaxIter;
    numUpdate = dfNumUpdate;
    stepSize = dfStepSize;
elseif nargin < 4
    numUpdate = dfNumUpdate;
    stepSize = dfStepSize;
elseif nargin < 5
    stepSize = dfStepSize;
end

dim = size(xTr, 1); % dimension of data
par.debug = 1; % show progress or not

par.ddIter = 20; % # iterations to keep trace for objective values 

par.b = ones(dim, 1)/dim; % initialized value for pseudo-count vector
% par.L = eye(dim); % initialized linear transform matrix
par.L = ilrInitL(dim); % some other initialized value for L (please refer to InitLOption folder)

par.K = 3; % number of targer neighbors

par.maxIter = maxIter; % number of maximum iterations in projected subgradient descent

par.theta = 1; % weight for pushing samples having different labels.
par.lambdaB = size(xTr, 2)*par.K/2; % regularizer for pseudo-count vector. (balancing value with 2 other objective components: pull & push ones)

par.stepSize = stepSize; % step size for PSGD
par.numUpdate = numUpdate; % # iterations for updating target neighbors

par.bEps = 1e-20; % For AITCHISON (log function): an offset value for projecting into positive orthant
par.thresholdSTOP = 1e-3; % threshold for stopping condition (relative different value of the objective one)
par.flagNORM = 1; % normalized for gradient (to adapt with the size of a dataset) 
par.flagNormGrad = 1; % normalized for gradient (to adapt with the size of a dataset)

iIter = 1; % current iteration
stopSG = 0; % stopping flag

% initial values for L, b !!!
bCur = par.b;
LCur = par.L;
QCur = LCur'*LCur; 

% for Nesterov's acceleration
bPre = bCur;
LPre = LCur;

% Using current L, b to generate Pair and Triplet
% log-mapping for training data
logxTr = log(xTr + repmat(bCur, 1, size(xTr, 2)));

% gen Pair & Triplet (N - KN - N/2)
[Det.gen, Det.NN, Det.DD] = aitGenDS(LCur, logxTr, yTr, par.K);   
idPair = Det.gen;
idTriplet = GenIdTriplet(Det.NN, Det.DD);
% idTriplet = GenIdTriplet(Det.NN, Det.DD, par.K, par.K);

flagStepSize = 0; % flag of adapting step size
condStepSizeL = 1; % adapt step size for dataset
condStepSizeb = 1; % adapt step size for dataset

% grad for Pair (w.r.t Q & b) & val
[gradQ1, gradb1] = AIT_GradientPairs(xTr, idPair, QCur, bCur, par.flagNORM);
[gradQ2, gradb2] = AIT_GradientTriplets(xTr, idTriplet, QCur, bCur, LCur, par.flagNORM);

if(par.flagNORM == 1)   % adapt for each objective term
    gradb3 = 2*bCur;    

    gradQ = gradQ1 + gradQ2;
    gradb = gradb1 + gradb2 + gradb3;
else                    % using the original setup (harder for choosing step size)
    gradb3 = 2*par.lambdaB*bCur;
    
    gradQ = gradQ1 + par.theta*gradQ2;
    gradb = gradb1 + par.theta*gradb2 + gradb3;
end

valCur = inf;

% save the best ones (L, b, #Iter, value of objective function)
LBest = LCur;
bBest = bCur;
tBest = iIter;
valBest = valCur;

% Projected subgradient descent
while (stopSG == 0)
    
    if par.debug == 1
        disp(['...........SG : ' num2str(iIter)]);
        disp(['...Current Objective Value: ' num2str(valCur)]);
    end
    
    % gradient of L w.r.t Q
    gradL = 2*LCur*gradQ;
    
    % adaptive step-size to unit gradient 
    if (par.flagNormGrad == 1)
       
        % save the first (keep it in one epoch before changing target neighbors)
        if (flagStepSize == 0)
            flagStepSize = 1;
            
            % adapted value
            condStepSizeL = norm(gradL, 'fro');
            condStepSizeb = norm(gradb, 2);           
        end
        
        % apply adapted step-size
        gradL = gradL ./ condStepSizeL;
        gradb = gradb ./ condStepSizeb;
                
    end
    
    % building momentum for Nesterov's acceleration
    nesCoef = (iIter - 2)/(iIter + 1);
    delb = nesCoef*(bCur - bPre);
    delL = nesCoef*(LCur - LPre);
    
    % update pseudo count vector with Nesterov's acceleration
    bNew = bCur + delb - (par.stepSize / sqrt(iIter)) * gradb;    
    % project b_new into offset of positive orthant R+
    bNew = max(bNew, par.bEps*ones(dim, 1));
    
    % update linear transformation matrix
    LNew = LCur + delL - (par.stepSize / sqrt(iIter)) * gradL;    
    QNew = LNew'*LNew;

    % calculate the new objective value
    valNew = AIT_ValObjFunc(xTr, idPair, idTriplet, QNew, bNew, LNew, par.theta, par.lambdaB);
    
    % compare between current objective value with the current best one
    if ((valBest - valNew)/valBest) <= par.thresholdSTOP
        % checking # epoch with ddIter (# iteration kept track) !!!
        if (iIter + 1 - tBest) > par.ddIter
            stopSG = 1; % stop the algorithm
            continue;    
        end
    else
        % update the best one
        LBest = LNew;
        bBest = bNew;
        tBest = iIter + 1;
        valBest = valNew;
    end
    
    % update current objective value
    valCur = valNew;

    % keep track for Nesterov's acceleration
    bPre = bCur;
    LPre = LCur;

    bCur = bNew;
    LCur = LNew;

    iIter = iIter + 1;

    % check maximum iteration for stopping
    if (iIter >= par.maxIter)
        stopSG = 1;
        continue;
    end

    %===============================
    % UPDATE NEAREST NEIGHBOR !!!
    if(mod(iIter, par.numUpdate) == 0)

        % DEBUG
        if(par.debug == 1)
            disp(['UPDATE NN for OPTIMIZING b !!!']);
        end

        % Generate new pair and triplet
        logxTr = log(xTr + repmat(bCur, 1, size(xTr, 2)));

        % gen Pair & Triplet (N - KN - N/2)
        [Det.gen, Det.NN, Det.DD] = aitGenDS(LCur, logxTr, yTr, par.K);   
        idPair = Det.gen;

        idTriplet = GenIdTriplet(Det.NN, Det.DD, par.K, par.K);

        flagStepSize = 0;
        condStepSizeL = 1;
        condStepSizeb = 1;

        % update current objective value
        QCur = LCur'*LCur;
        valCur = AIT_ValObjFunc(xTr, idPair, idTriplet, QCur, bCur, LCur, par.theta, par.lambdaB);

    end
    %===============================

    % UPDATE gradQ & gradb at new L & b
    % Building momentum for Nesterov's acceleration
    nesCoef = (iIter - 2)/(iIter + 1);
    bb = bCur + nesCoef*(bCur - bPre);
    
    % keep pseudo-count in the positive orthant
    if sum(bb < 0) > 0
        bb = max(bb, par.bEps*ones(dim, 1));
    end
    
    LL = LCur + nesCoef*(LCur - LPre);
    QQ = LL'*LL;

    % Calculate gradients for pairs and triplets w.r.t L & b
    [gradQ1, gradb1] = AIT_GradientPairs(xTr, idPair, QQ, bb, par.flagNORM);
    [gradQ2, gradb2] = AIT_GradientTripletsW(xTr, idTriplet, QQ, bb, par.flagNORM);

    if(par.flagNORM == 1) % adapt for each objective term
        gradb3 = 2*bb;
        
        gradQ = gradQ1 + gradQ2;
        gradb = gradb1 + gradb2 + gradb3;
    else                  % using the original setup (harder for choosing step size)
        gradb3 = 2*par.lambdaB*bb;
        
        gradQ = gradQ1 + par.theta*gradQ2;
        gradb = gradb1 + par.theta*gradb2 + gradb3;
    end
    
end
end


