#include "mex.h"
/*
 * #include "omp.h"
*/
/*
 
 mex SOGD.c COPTIMFLAGS="-fopenmp \$COPTIMFLAGS" LDOPTIMFLAGS="-fopenmp \$LDOPTIMFLAGS"

 */

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  /* Declare variables. */ 

  double *X, *v1,*v2, *C;
  double *av,*bv;
  
  int m,n,inds;
  int i,r,c,j;
  
      
  /* Get the number of elements in the input argument. */  
  inds = mxGetNumberOfElements(prhs[1]);

  n = mxGetN(prhs[0]);
  m = mxGetM(prhs[0]);
  
  /* Get the data. */
  X  = mxGetPr(prhs[0]);
  av  = mxGetPr(prhs[1]);
  bv  = mxGetPr(prhs[2]);

  /* Create output matrix */
  plhs[0]=mxCreateDoubleMatrix(m,m,mxREAL);
  C=mxGetPr(plhs[0]);

  /* for all pair */
  /*
    #pragma omp parallel reduction(+: C)
  */
/*  
#pragma omp for schedule(dynamic)
  
 */
  /*#pragma omp parallel for*/
  for(i=0;i<inds;i++){

   /* Assign cols addresses */
   v1=&X[(int) (av[i]-1)*m];
   v2=&X[(int) (bv[i]-1)*m];
   
   j = 0;
   for(c=0;c<m;c++){
       for(r=0;r<m;r++){
         C[j++] += v1[r]*v2[c];
      }
   }    
  
  }

}



