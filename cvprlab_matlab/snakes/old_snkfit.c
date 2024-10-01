/**********************************************************\
SNKFIT.C  --   Matlab MEX implementation of the Williams/
               Shah snake relaxation algorithm for Catmull-
							 Rom splines

               John Collomosse 2003

\**********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mex.h"

#define TRUE            (-1)
#define FALSE           (0)

#define NEIGHBOURHOOD   (5)            /* Half neighbourhood size for search */
#define ALPHA           (1.0)            /* Alpha weight (1st deriv, internal) */
#define BETA            (1.1)          /* Beta weight (2nd deriv, internal)  */
#define GAMMA           (1.2)          /* Gamma weight (Image data, external)*/
#define IMAGELOWLIMIT   (15.0/255.0)   /* Below this, pixels are equiv to 0  */
#define MAXITER         (50)           /* Max iterations for relaxation      */
#define SNK_DEBUG       (1)            /* Value is irrelevant. If defined
                                          will output progress to stdout     */

/* relax corner constraint if img magnitude > 100 and cos(bendangle)>0.25 (30 deg) */
#define RELAX_CORNER_IMG (200.0/255.0)
#define RELAX_CORNER_BETA (0.25)

#define NEIGHBSQ        ((NEIGHBOURHOOD*2+1)*(NEIGHBOURHOOD*2+1))

typedef struct _tuple {
        double alpha;
        double beta;
        double gamma;
} tuple;

int     get_params(const mxArray*[], int, double**, int*, int*, double**, int*, int*);
void    calc_energy_circ(double*,int,double*,int,int,double*,double*,double*);
void    calc_energy_lin(double*,int,double*,int,int,double*,double*,double*);


static double PI;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

	double*		salImage;
        double*         ptr;
        int             height,width;
        int             iter;
	mxArray*	ansarray;
        int             dims[2];
        double          energy;
        double*         pts;
        double*         newpts;
        int             pts_enum;
        int             ptsmoved;
        int             i,x,y,tmp;
        tuple*          tuples;
        double*         gridA;
        double*         gridB;
        double*         gridC;
        double*         curvey;
        double          a,b,c,maxA,maxB,maxC,minC,maxEval,maxEidx;
        double          thisp_x, thisp_y;
        double          nextp_x, nextp_y;
        double          prevp_x, prevp_y;
        double          dst1_x,dst1_y,dst2_x,dst2_y;
        double          dotted;
        int             circular;
       
	PI=atan(1.0)*4.0;
        

        if (!get_params(prhs, nrhs, &salImage, &height, &width, &pts, &pts_enum, &circular)) {
                mxErrMsgTxt("Bad command line parameters\nUse SNKFIT <image field> <init snakeloop controlpoints (1:2,n)> <circular(0) or linear(1) snake>");
	}

        /* Setup tuples */
        tuples=mxCalloc(pts_enum,sizeof(tuple));
        for (i=0; i<pts_enum; i++) {
                tuples[i].alpha=ALPHA;
                tuples[i].beta=BETA;
                tuples[i].gamma=GAMMA;
        }

        newpts=(double*)mxCalloc(pts_enum*2,sizeof(double));
        gridA=(double*)mxCalloc(NEIGHBSQ,sizeof(double));
        gridB=(double*)mxCalloc(NEIGHBSQ,sizeof(double));
        gridC=(double*)mxCalloc(NEIGHBSQ,sizeof(double));
        curvey=(double*)mxCalloc(pts_enum,sizeof(double));

        energy=0;
        /* Iterate snake */
        ptsmoved=1;
        iter=0;
        while (ptsmoved>0 && iter<MAXITER) {

#ifdef SNK_DEBUG
                printf("Iteration %d\n",iter+1);
#endif
                ptsmoved=0;
                for (i=0; i<pts_enum; i++) {
                        for (tmp=0; tmp<(NEIGHBOURHOOD*2+1)*(NEIGHBOURHOOD*2+1); tmp++) {
                                *(gridA+tmp)=0;
                                *(gridB+tmp)=0;
                                *(gridC+tmp)=0;
                        }
                        for (x=-NEIGHBOURHOOD; x<=NEIGHBOURHOOD; x++) {
                                for (y=-NEIGHBOURHOOD; y<=NEIGHBOURHOOD; y++) {
                                        for (tmp=0; tmp<pts_enum*2; tmp++)
                                                *(newpts+tmp)=*(pts+tmp);
                                        *(newpts+(2*i))+=(double)x;
                                        *(newpts+1+(2*i))+=(double)y;
                                        if (circular==0) {
                                                calc_energy_circ(newpts,pts_enum,salImage,width,height,&a,&b,&c);
                                        }
                                        else {
                                                calc_energy_lin(newpts,pts_enum,salImage,width,height,&a,&b,&c);
                                        }
                                        *(gridA+(x+NEIGHBOURHOOD)+((y+NEIGHBOURHOOD)*(NEIGHBOURHOOD*2+1)))=a;
                                        *(gridB+(x+NEIGHBOURHOOD)+((y+NEIGHBOURHOOD)*(NEIGHBOURHOOD*2+1)))=b;
                                        *(gridC+(x+NEIGHBOURHOOD)+((y+NEIGHBOURHOOD)*(NEIGHBOURHOOD*2+1)))=c;
                                }
                        }
                        /* normalise A B and C */
                        maxA=*gridA;
                        maxB=*gridB;
                        minC=maxC=*gridC;                                        
                        for (tmp=0; tmp<NEIGHBSQ; tmp++) {
                              if (maxA<*(gridA+tmp))
                                   maxA=*(gridA+tmp);
                              if (maxB<*(gridB+tmp))
                                   maxB=*(gridB+tmp);
                              if (maxC<*(gridC+tmp))
                                   maxC=*(gridC+tmp);
                              if (minC>*(gridC+tmp))
                                   minC=*(gridC+tmp);
                        }
                        if ((maxC-minC)<IMAGELOWLIMIT)
                              minC=maxC-IMAGELOWLIMIT;

                        maxEidx=-1;
                        maxEval=0;
                        for (tmp=0; tmp<NEIGHBSQ; tmp++) {
                              a=*(gridA+tmp);
                              if (maxA!=0)
                                  a/=maxA;
                              b=*(gridB+tmp);
                              if (maxB!=0)
                                  b/=maxB;
                              c=(*(gridC+tmp)-minC)/(maxC-minC);
                              energy=-(tuples[i].alpha*a)-tuples[i].beta*b+(tuples[i].gamma*c);
                              if (energy>maxEval || maxEidx==-1) {
                                  maxEidx=tmp;
                                  maxEval=energy;
                              }
                        }
                        /* we now know which point to move to (maxEidx) */
                        y=(int)floor(maxEidx/(NEIGHBOURHOOD*2.0+1.0));
                        x=(int)maxEidx%(NEIGHBOURHOOD*2+1);
                        y=y-NEIGHBOURHOOD;
                        x=x-NEIGHBOURHOOD;

                        if (!(x==0 && y==0)) {
                                /* point moved */
                                *(pts+(2*i))+=(double)x;
                                *(pts+1+(2*i))+=(double)y;
                                ptsmoved++;
                        }
                        
                                        
                }

                /* relax the corner constraint */
                for (i=0; i<pts_enum; i++) {
        
                   thisp_x=*(pts+(2*i));
                   thisp_y=*(pts+1+(2*i));
                   nextp_x=*(pts+(2*((i+1)%pts_enum)));
                   nextp_y=*(pts+1+(2*((i+1)%pts_enum)));
                   if (i>0) {
                           prevp_x=*(pts+(2*(i-1)));
                           prevp_y=*(pts+1+(2*(i-1)));
                   }
                   else {
                           prevp_x=*(pts+(2*(pts_enum-1)));
                           prevp_y=*(pts+1+(2*(pts_enum-1)));
                   }

                   dotted=sqrt(((nextp_x-thisp_x)*(nextp_x-thisp_x))+
                              ((nextp_y-thisp_y)*(nextp_y-thisp_y)));
                   dst1_x=(nextp_x-thisp_x)/dotted;
                   dst1_y=(nextp_y-thisp_y)/dotted;
                   dotted=sqrt(((thisp_x-prevp_x)*(thisp_x-prevp_x))+
                              ((thisp_y-prevp_y)*(thisp_y-prevp_y)));
                   dst2_x=(thisp_x-prevp_x)/dotted;
                   dst2_y=(thisp_y-prevp_y)/dotted;
                   dotted=(dst1_x*dst2_x)+(dst1_y*dst2_y);

                   *(curvey+i)=dotted;
                        
                }
                for (i=circular; i<pts_enum-circular; i++) {
        
                   thisp_x=*(curvey+(2*i));
                   nextp_x=*(curvey+(2*((i+1)%pts_enum)));
                   if (i>0) {
                           prevp_x=*(curvey+(2*(i-1)));
                   }
                   else {
                           prevp_x=*(curvey+(2*(pts_enum-1)));
                   }

                   if (thisp_x>prevp_x && thisp_x>nextp_x) {
                        if (thisp_x>RELAX_CORNER_BETA) {
                                a=*(salImage+((int)floor(*(pts+1+(2*i))))+(int)floor((*(pts+(2*i)))*height));
                                if (a>RELAX_CORNER_IMG) {
                                        tuples[i].beta=0;
                                }
                        }
                   }
                }
                iter++;
        }  

#ifdef SNK_DEBUG
        if (ptsmoved>0)
                printf("Non-convergence within %d iterations\n",MAXITER);
        else
                printf("Converged in %d iterations\n",iter);
#endif


        /* Return control points of fitted spline to matlab */

        printf("Returning\n");

        dims[0]=2;
        dims[1]=pts_enum;
        ansarray=mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
        ptr=mxGetData(ansarray);
        *ptr=energy;
        for (i=0; i<pts_enum*2; i++) {
                *(ptr+i)=*(pts+i);
        }

        plhs[0]=ansarray;

        mxFree(tuples);
        mxFree(newpts);
        mxFree(gridA);
        mxFree(gridB);
        mxFree(gridC);
        mxFree(curvey);
}

int get_params(const mxArray* args[], int argc, double** salImage, int* height, int* width, double** ptsarray, int* ptscount, int* circular) {

	const int* dims;
        int     i;

        if (argc!=3) {
                return FALSE;
        }

        /* Get double energy field */
        if (!mxIsDouble(args[0]) || mxIsComplex(args[0]))
		return FALSE;	
	else {
                dims=mxGetDimensions(args[0]);
                *height=dims[0];
                *width=dims[1];
                *salImage=mxGetPr(args[0]);
	}
                       
                                                 


        /* Snake points */
        if (!mxIsDouble(args[1]) || mxIsComplex(args[1]))
		return FALSE;	
	else {
                dims=mxGetDimensions(args[1]);
                *ptscount=dims[1];

                if (dims[1]%2==1 || dims[1]<4) {
                        mxErrMsgTxt("Specify 2 x n matrix of control points\nwhere 'n' is even, and >=4");
		}
		else {
                        *ptsarray=mxCalloc(*ptscount,2*sizeof(double));
                        for (i=0; i<(*ptscount)*2; i++)
                                *((*ptsarray)+i)=*(mxGetPr(args[1])+i);
		}
	}

        /* Circular */
        if (!mxIsDouble(args[2]) || mxIsComplex(args[2]))
		return FALSE;	
	else {
                dims=mxGetDimensions(args[2]);

                if (dims[1]!=1 || dims[0]!=1) {
                        mxErrMsgTxt("Specify 0 for a circular snake (start and end join up), or 1 for a linear one");
		}
		else {
                        if (*mxGetPr(args[2])==0)
                                *circular=0;
                        else
                                *circular=1;
		}
	}

	return TRUE;

}




void    calc_energy_circ(double* pts,int pts_enum,double* field,int width,int height,double* a,double* b,double* c) {

        int i;
        int x,y;
        double xfull,yfull,next_x,next_y;
        double thisp_x,thisp_y,prevp_x,prevp_y,nextp_x,nextp_y;
        double dst1_x,dst1_y,dst2_x,dst2_y,dotted;
        double avgdist;

        *c=0; /* image energy */
        avgdist=0;
        for (i=0; i<pts_enum; i++) {
                xfull=(*(pts+(2*i)));
                yfull=(*(pts+1+(2*i)));
                x=(int)xfull;
                y=(int)yfull;
                if (x>=0 && x<width && y>=0 && y<height) {
                        (*c)+=*(field+y+(x*height));
                }
                next_x=*(pts+2*((i+1)%pts_enum));
                next_y=*(pts+1+2*((i+1)%pts_enum));
                avgdist+=sqrt(((next_x-xfull)*(next_x-xfull))+
                         ((next_y-yfull)*(next_y-yfull)));
        }
        avgdist/=pts_enum;

        *a=0;
        for (i=0; i<pts_enum; i++) {
                xfull=(*(pts+(2*i)));
                yfull=(*(pts+1+(2*i)));
                next_x=*(pts+2*((i+1)%pts_enum));
                next_y=*(pts+1+2*((i+1)%pts_enum));
                (*a)+=fabs(avgdist-( sqrt( ((xfull-next_x)*(xfull-next_x))+((yfull-next_y)*(yfull-next_y)) ) ));
        }        

        *b=0;
        for (i=0; i<pts_enum; i++) {
       
             thisp_x=*(pts+(2*i));
             thisp_y=*(pts+1+(2*i));
             nextp_x=*(pts+(2*((i+1)%pts_enum)));
             nextp_y=*(pts+1+(2*((i+1)%pts_enum)));
             if (i>0) {
                  prevp_x=*(pts+(2*(i-1)));
                  prevp_y=*(pts+1+(2*(i-1)));
             }
             else {
                  prevp_x=*(pts+(2*(pts_enum-1)));
                  prevp_y=*(pts+1+(2*(pts_enum-1)));
             }

             dotted=sqrt(((nextp_x-thisp_x)*(nextp_x-thisp_x))+
                        ((nextp_y-thisp_y)*(nextp_y-thisp_y)));
             dst1_x=(nextp_x-thisp_x)/dotted;
             dst1_y=(nextp_y-thisp_y)/dotted;
             dotted=sqrt(((thisp_x-prevp_x)*(thisp_x-prevp_x))+
                        ((thisp_y-prevp_y)*(thisp_y-prevp_y)));
             dst2_x=(thisp_x-prevp_x)/dotted;
             dst2_y=(thisp_y-prevp_y)/dotted;
             dotted=(dst1_x*dst2_x)+(dst1_y*dst2_y);
             (*b)=1-dotted;
                        
        } 

}

void    calc_energy_lin(double* pts,int pts_enum,double* field,int width,int height,double* a,double* b,double* c) {

        int i;
        int x,y;
        double xfull,yfull,next_x,next_y;
        double thisp_x,thisp_y,prevp_x,prevp_y,nextp_x,nextp_y;
        double dst1_x,dst1_y,dst2_x,dst2_y,dotted;
        double avgdist;

        *c=0; /* image energy */
        avgdist=0;
        for (i=0; i<pts_enum; i++) {
                xfull=(*(pts+(2*i)));
                yfull=(*(pts+1+(2*i)));
                x=(int)xfull;
                y=(int)yfull;
                if (x>=0 && x<width && y>=0 && y<height) {
                        (*c)+=*(field+y+(x*height));
                }
                if (i!=pts_enum-1) {
                        next_x=*(pts+2*((i+1)));
                        next_y=*(pts+1+2*((i+1)));
                        avgdist+=sqrt(((next_x-xfull)*(next_x-xfull))+
                                 ((next_y-yfull)*(next_y-yfull)));
                }
        }
        avgdist/=(pts_enum-1);

        *a=0;
        for (i=0; i<pts_enum-1; i++) {
                xfull=(*(pts+(2*i)));
                yfull=(*(pts+1+(2*i)));
                next_x=*(pts+2*((i+1)%pts_enum));
                next_y=*(pts+1+2*((i+1)%pts_enum));
                (*a)+=fabs(avgdist-( sqrt( ((xfull-next_x)*(xfull-next_x))+((yfull-next_y)*(yfull-next_y)) ) ));
        }        

        *b=0;
        for (i=1; i<pts_enum-1; i++) {
       
             thisp_x=*(pts+(2*i));
             thisp_y=*(pts+1+(2*i));
             nextp_x=*(pts+(2*((i+1)%pts_enum)));
             nextp_y=*(pts+1+(2*((i+1)%pts_enum)));
             if (i>0) {
                  prevp_x=*(pts+(2*(i-1)));
                  prevp_y=*(pts+1+(2*(i-1)));
             }
             else {
                  prevp_x=*(pts+(2*(pts_enum-1)));
                  prevp_y=*(pts+1+(2*(pts_enum-1)));
             }

             dotted=sqrt(((nextp_x-thisp_x)*(nextp_x-thisp_x))+
                        ((nextp_y-thisp_y)*(nextp_y-thisp_y)));
             dst1_x=(nextp_x-thisp_x)/dotted;
             dst1_y=(nextp_y-thisp_y)/dotted;
             dotted=sqrt(((thisp_x-prevp_x)*(thisp_x-prevp_x))+
                        ((thisp_y-prevp_y)*(thisp_y-prevp_y)));
             dst2_x=(thisp_x-prevp_x)/dotted;
             dst2_y=(thisp_y-prevp_y)/dotted;
             dotted=(dst1_x*dst2_x)+(dst1_y*dst2_y);
             (*b)=1-dotted;
                        
        } 

}
