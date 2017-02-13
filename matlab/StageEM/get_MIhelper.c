/*
Input raw data (n samples) and co-occurance matrix. rearrange co-occurance matrix according to raw data. output should be m by m by n.

*/
#include "mex.h"
#include <math.h>


void MIhelper(mwSize m,mwSize n,mwSize k,double* z,double *CO,double *output){
  /* z (k by n by m) is the raw data.
   CO (m by m by k by k) is the co-occurance matrix
   output (m by m by n) is the output */
  mwSize j,a,b,l,l1,l2;
  double eps = 0.00000001;
  /*Check the sizes of inputs and output*/
  /*if(sizeof(z)/sizeof(z[0])!=k*n*m) /* check z */
  /*  return;
  /* if(sizeof(CO)/sizeof(CO[0])!=m*m*k*k) /* check CO */
  /*  return;
  /*  if(sizeof(output)/sizeof(output[0])!=m*m*n)
      return;*/
  
  /*Do the computation*/
  for(j=0;j<n;j++){
    for(a=0;a<m;a++){
      for(l=0;l<k;l++){
	if(abs(z[l+j*k+a*k*n]-1)<eps){
	  l1 = l;
	  break;
	}
      }
      for(b=0;b<m;b++){
	for(l=0;l<k;l++){
	  if(abs(z[l+j*k+b*k*n]-1)<eps){
	    l2 = l;
	    break;
	  }
	}
	output[a+b*m+j*m*m] = log(CO[a+b*m+l1*m*m+l2*m*m*k]);
      }
    }
  }
  return;
}

/* The gateway function*/
/*************************
prhs[0] = z;
prhs[1] = CO;
************************/

void mexFunction(int nlhs,mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  if(nrhs != 2){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","Two inputs required");
  }
  if(nlhs != 1){
    mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required");
  }
  mwSize m,n,k; /* size of z */
  const mwSize *foo;
  double *z; /* to represent plhs[0] k by n by m array */
  double *CO; /* to represent plhs[1] m by m by k by k array */
  double *output;
  mwSize OutDim[3]; 
  /* get Input dimension for plhs[0] (which is z) */
  foo = mxGetDimensions(prhs[0]);
  k = foo[0];
  n = foo[1];
  m = foo[2];
  z = mxGetPr(prhs[0]);
  CO = mxGetPr(prhs[1]);
  /* Prepare output */
  OutDim[0] = m;
  OutDim[1] = m;
  OutDim[2] = n;
  plhs[0] = mxCreateNumericArray(3,OutDim,mxDOUBLE_CLASS,mxREAL);
  output = mxGetPr(plhs[0]);
  /* Perform calculation */
  MIhelper(m,n,k,z,CO,output);
}

