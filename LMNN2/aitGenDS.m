% generate for similar set & dissimilar set

function [gen, SS, DD] = aitGenDS(L,x,y,Kg)

% input:
% x: dim x n
% y
% Kg: number of NN
%
% output:
% [gen, SS]: for similar labels
% DD: for different labels

XX = sqdistance(x, x, [], L);

[gen, SS] = poolGenLS(XX, x, y, Kg);
DD = poolGenLD(XX, x, y, Kg);

function [gen,NN]=poolGenLS(XX, x, y, Kg);

[~,N]=size(x);

un=unique(y);
Gnn=zeros(Kg,N);

for c=un
 fprintf('%i nearest MY neighbors for class %i:',Kg,c);
 i=find(y==c);

 % note: i - index for computation !!! 
 nn = LSKnnPoolS(XX(i, i), length(i), 2:Kg+1);
 
 Gnn(:,i)=i(nn);
 
 clear i;
 clear nn;
 
 fprintf('\r');
end;

fprintf('\n');
NN=Gnn;

gen1=vec(Gnn(1:Kg,:)')';
gen2=vec(repmat(1:N,Kg,1)')';
gen=[gen1;gen2];

clear('gen1', 'gen2', 'Gnn', 'un');

function NN=LSKnnPoolS(XX, sizeX2, ks);

N=sizeX2;
NN=zeros(length(ks),N);

B = N;

for i=1:B:N

  BB=min(B,N-i);
  fprintf('.');
  
  Dist = XX(:, i:i+BB);  
  [~, nn]=mink(Dist,max(ks));
  clear('Dist');

  NN(:,i:i+BB)=nn(ks,:);
  clear('nn');
  
end;
  
function NN=poolGenLD(XX, x, y,Kg)

[~,N]=size(x);

NN = zeros(Kg, N);

fprintf('%i nearest MY neighbors for different classes \n',Kg);

for ii=1:N
    
    % id of different class
    if(mod(ii, 100) == 0)
        fprintf('.');
        if(mod(ii, 1000) == 0)
            fprintf('\n');
        end
    end
    
    id = find(y ~= y(ii));   
    NN(:, ii) = id(LSKnnPoolD(XX(id, ii), 1, 1:Kg));
    clear id;
    
end
fprintf('\n');


function NN=LSKnnPoolD(XX, sizeX2, ks);

N=sizeX2;
NN=zeros(length(ks),N);

B=N;
for i=1:B:N

  BB=min(B,N-i);
  fprintf('.');
  
  Dist = XX(:, i:i+BB);
  [~, nn]=mink(Dist,max(ks));  
  clear('Dist');
  
  NN(:,i:i+BB)=nn(ks,:);
  clear('nn');
  
end;
