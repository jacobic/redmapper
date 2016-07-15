#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "solver_nfw.h"

int solver_nfw(double r0, double beta, long ngal,
	       double *ucounts, double *bcounts, double *r, double *w,
	       double *lambda, double *p, double *wt, double tol,
	       double *cpars, double rsig)
{
  double lamlo,lamhi,mid,outlo,outmid,rc;
  double cval;
  int i;

  lamlo=0.5;
  lamhi=2000.0;
  outlo=-1.0;
  
  while (fabs(lamhi-lamlo) > 2*tol) {
    mid=(lamhi+lamlo)/2.0;
    if (outlo < 0.0) {
      nfw_weights(lamlo,r0,beta,ngal,ucounts,bcounts,r,w,p,wt,&rc,rsig);
      outlo=0.0;
      for (i=0;i<ngal;i++) {
	outlo+=wt[i];
      }
      cval = cpars[0] + cpars[1]*rc + cpars[2]*rc*rc + cpars[3]*rc*rc*rc;
      //if (cval < 0.0) { cval = 0.0; }
      
      outlo += lamlo*cval;
    }
    nfw_weights(mid,r0,beta,ngal,ucounts,bcounts,r,w,p,wt,&rc,rsig);
    outmid=0.0;
    for (i=0;i<ngal;i++) {
      outmid+=wt[i];
    }
    cval = cpars[0] + cpars[1]*rc + cpars[2]*rc*rc + cpars[3]*rc*rc*rc;
    //if (cval < 0.0) { cval = 0.0; }
    
    outmid += mid*cval;
    
    if (outlo < 1.0) { outlo = 0.9;} // stability at low end
    if ((outlo-lamlo)*(outmid-mid) > 0.0) {
      lamlo=mid;
      outlo=-1.0;
    } else {
      lamhi=mid;
    }
  }

  *lambda = (lamlo+lamhi)/2.0;

  if (*lambda < 1.0) {
    *lambda = -1.0;
  }
  
  return 0;
}