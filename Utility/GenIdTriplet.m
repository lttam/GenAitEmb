function id = GenIdTriplet(idNN, idDD, kS, kD)
% INPUT:
% idNN: k x n (k nearest neighbor for each sample - same class label)
% idDD: k x n (k nearest neighbor for each sample - different class label)
% kS: number of same class (used)
% kD: number of different class (used)

% OUTPUT:
% id: 3 x nTriplet
% where (I, J, K) s.t (I, J): same class & (I, K): different class

% generate all Triplets from nearest neighbors and their labels, and all the
% others.

% from nearest neighbors --> gen pairs
% form pairs, add other different label samples --> Triplets

% dfD = size(idDD, 1);
dfD = 1;

if(nargin < 3)
    kS = size(idNN, 1);
    kD = dfD;
elseif(nargin < 4)
    kD = dfD;
end

nSamples = size(idNN, 2);

upperBoundNTriplet = nSamples * kS * kD;
id = zeros(3, upperBoundNTriplet);

nTriplet = 0;

for kI = 1:nSamples
    for kJ = 1:kS
        % pair kkI, kkJ
        for kK = 1:kD
            nTriplet = nTriplet + 1;
            id(:, nTriplet) = [kI ; idNN(kJ, kI) ; idDD(kK, kI)];       
        end
    end
end

id = id(:, 1:nTriplet);

% % % keep random the same for each iteration !!!
% % rand('seed', 1);
% % 
% % % set new nTriplet for randomly sampling !!!
% % nPer = 0.8; % only keep 80% triplets !!!
% % nTriplet = nTriplet * nPer;
% % chosenTriplet = min(nTriplet, nSamples * kNeighbors * kNeighbors);
% % idRandTriplet = randperm(nTriplet);
% % 
% % id = id(:, idRandTriplet(1:chosenTriplet));

end
