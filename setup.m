% Add a modified LMNN solver of Weinberger
% http://www.cse.wustl.edu/~kilian/code/page21/page21.html

cd Utility
mex SOGD.c
addpath(pwd)
cd ..

cd InitLOption
addpath(pwd)
cd ..

cd LMNN2/
setpaths
cd ..
