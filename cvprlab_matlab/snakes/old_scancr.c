/******************************************************\
SCANCR.C  --   Matlab MEX function to scan convert a
               Catmull-Rom spline

               John Collomosse 2002

\******************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mex.h"

#define TRUE            (-1)
#define FALSE           (0)

#define SBLOCK_D        (0.01)
#define SBLOCK_INC      (0.001)

typedef struct sblock_entry_struct {

	int entry_enum;
	double* idx;
	double* dist;	

} SBLOCK_ENTRY;

typedef struct sblock_struct {

	double distance;
	int    entry_enum;
	SBLOCK_ENTRY* entries;

} SBLOCK;

int     get_params(const mxArray*[], int, int*, int*, double**, int*, int*);

static double PI;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

	double*		canvas;
	double*   ptr;
  int       height,width;
  int       thickness,ptsenum,i;
  double*   pts;
	mxArray*	ansarray;
  int       dims[2];
	double    s,c,r;
	double    interval;
	double*   GBLOCK;
	SBLOCK    theSBLOCK;
	double*   G;
	double    g1x,g1y,g2x,g2y,g3x,g3y,g4x,g4y,px,py,q1,q2,q3,q4;
	double    Prx,Pry,Psx,Psy;
	int       tmpentries,tmplp;
	double*   tmpdist;
	double*   tmpidx;
	double*   scratch;
	double    diff,diffx,diffy;
	double    distvar,themin,themax,cumtot,minval,maxval;
	int       ptlp,sfunc,minidx,maxidx,mskx,msky,lpx,lpy,maskoffset;
	double    minic,min_s,max_s,ratio;
	double*      maskblock;

  PI=atan(1.0)*4.0;
        
  if (!get_params(prhs, nrhs, &height, &width, &pts, &ptsenum, &thickness)) {
		mxErrMsgTxt("Scan converts a Catmull-Rom spline, given the control points and a thickness of spline, and a raster size\nSCANCR:  Use SCANCR <canvas width> <canvas height> <spline points (1:2,1:n) n is num points>=4> <thickness>");
	}

	/* Create canvas */
	canvas=(double*)mxCalloc(height*width,sizeof(double));
	maskoffset=thickness;
	maskblock=(double*)mxCalloc((thickness*2+1)*(thickness*2+1),sizeof(double));
	if (!canvas || !maskblock) {
		mxErrMsgTxt("SCANCR:  Out of memory creating canvas for CR spline scan conversion");
	}
	
	for (lpx=-maskoffset; lpx<=maskoffset; lpx++) {
		for (lpy=-maskoffset; lpy<=maskoffset; lpy++) {
			distvar=sqrt((double)(lpx*lpx+lpy*lpy));
			if (distvar<=thickness) {
				mskx=lpx+maskoffset;
				msky=lpy+maskoffset;
				*(maskblock+msky+(mskx*(maskoffset*2+1)))=1;
			}
		}
	}
	

  /* Do the scan conversion of the spline */
	/*	printf("Scan converting spline with %d control points\n",ptsenum);*/

	/* 1) Get gblock - the gblock is xxxxx;yyyy;xxxxx;yyyy and so on 2xn n=num ctl pts minus 1*/
	/* based on p03_buildgblock */
	GBLOCK=(double*)mxCalloc(2*ptsenum*4,sizeof(double));
	G=(double*)mxCalloc(8,sizeof(double));
	for (i=0; i<ptsenum-1; i++) {
		if (i==0) {
			*(G)=*pts;
			*(G+1)=*(pts+1);
			*(G+2)=*pts;
			*(G+3)=*(pts+1);
			*(G+4)=*(pts+2);
			*(G+5)=*(pts+3);
			*(G+6)=*(pts+4);
			*(G+7)=*(pts+5);
		}
		else if (i==ptsenum-2) {
			*(G)=*(pts+((ptsenum-3)*2));
			*(G+1)=*(pts+((ptsenum-3)*2)+1);
			*(G+2)=*(pts+((ptsenum-2)*2));
			*(G+3)=*(pts+((ptsenum-2)*2)+1);
			*(G+4)=*(pts+((ptsenum-1)*2));
			*(G+5)=*(pts+((ptsenum-1)*2)+1);
			*(G+6)=*(pts+((ptsenum-1)*2));
			*(G+7)=*(pts+((ptsenum-1)*2)+1);

		}
		else {
			*(G)=*(pts+((i-1)*2));
			*(G+1)=*(pts+((i-1)*2)+1);
			*(G+2)=*(pts+((i)*2));
			*(G+3)=*(pts+((i)*2)+1);
			*(G+4)=*(pts+((i+1)*2));
			*(G+5)=*(pts+((i+1)*2)+1);
			*(G+6)=*(pts+((i+2)*2));
			*(G+7)=*(pts+((i+2)*2)+1);
		}

		/* copy G into GBLOCK */
		ptr=GBLOCK+(i*8);
	  *ptr=*G;
		*(ptr+1)=*(G+2);
		*(ptr+2)=*(G+4);
		*(ptr+3)=*(G+6);
	  *(ptr+4)=*(G+1);
		*(ptr+5)=*(G+3);
		*(ptr+6)=*(G+5);
		*(ptr+7)=*(G+7);

	}


	/* 2) Get sblock (based on p03_buildsblock) */
	theSBLOCK.distance=0;
	theSBLOCK.entry_enum=ptsenum-1;
	theSBLOCK.entries=(SBLOCK_ENTRY*)mxCalloc(theSBLOCK.entry_enum,sizeof(SBLOCK_ENTRY));
	for (i=0; i<theSBLOCK.entry_enum; i++) {

		tmpentries=0;
		tmpidx=NULL;
		tmpdist=NULL;

		g1x=*(GBLOCK+(i*8));
		g2x=*((GBLOCK+(i*8))+1);
		g3x=*((GBLOCK+(i*8))+2);
		g4x=*((GBLOCK+(i*8))+3);
		g1y=*((GBLOCK+(i*8))+4);
		g2y=*((GBLOCK+(i*8))+5);
		g3y=*((GBLOCK+(i*8))+6);
		g4y=*((GBLOCK+(i*8))+7);

 
		s=0;
		c=0;

		while (s<=1) {

			r=s;

			q1=(-0.5*r*r*r)+(1*r*r)+(-0.5*r)+0;
			q2=(1.5*r*r*r)+(-2.5*r*r)+(0*r)+1;
			q3=(-1.5*r*r*r)+(2*r*r)+(0.5*r)+0;
			q4=(0.5*r*r*r)+(-0.5*r*r)+(0*r)+0;

			Prx=g1x*q1 + g2x*q2 + g3x*q3 + g4x*q4;
			Pry=g1y*q1 + g2y*q2 + g3y*q3 + g4y*q4;

			/* expand lookup table */
			scratch=tmpdist;
			tmpdist=mxCalloc(tmpentries+1,sizeof(double));
			for (tmplp=0; tmplp<tmpentries; tmplp++) {
				tmpdist[tmplp]=scratch[tmplp];
			}			
			if (tmpentries>0)
				mxFree(scratch);
			scratch=tmpidx;
			tmpidx=mxCalloc(tmpentries+1,sizeof(double));
			for (tmplp=0; tmplp<tmpentries; tmplp++) {
				tmpidx[tmplp]=scratch[tmplp];
			}			
			if (tmpentries>0)
				mxFree(scratch);
			tmpentries++;
			tmpidx[tmpentries-1]=r;
			tmpdist[tmpentries-1]=c;

			/* inner loop */
			while (1) {
				s+=SBLOCK_INC;

				q1=(-0.5*s*s*s)+(1*s*s)+(-0.5*s)+0;
				q2=(1.5*s*s*s)+(-2.5*s*s)+(0*s)+1;
				q3=(-1.5*s*s*s)+(2*s*s)+(0.5*s)+0;
				q4=(0.5*s*s*s)+(-0.5*s*s)+(0*s)+0;
				
				Psx=g1x*q1 + g2x*q2 + g3x*q3 + g4x*q4;
				Psy=g1y*q1 + g2y*q2 + g3y*q3 + g4y*q4;

				diffx=Psx-Prx;
				diffy=Psy-Pry;
				diff=sqrt((diffx*diffx)+(diffy*diffy));
				if (diff-SBLOCK_D>0)
					break;			
			}
			
			c+=diff;
		}
		
		/* expand lookup table */
		scratch=tmpdist;
		tmpdist=mxCalloc(tmpentries+1,sizeof(double));
		for (tmplp=0; tmplp<tmpentries; tmplp++) {
			tmpdist[tmplp]=scratch[tmplp];
		}			
		if (tmpentries>0)
			mxFree(scratch);
		scratch=tmpidx;
		tmpidx=mxCalloc(tmpentries+1,sizeof(double));
		for (tmplp=0; tmplp<tmpentries; tmplp++) {
			tmpidx[tmplp]=scratch[tmplp];
		}			
		if (tmpentries>0)
			mxFree(scratch);
		tmpentries++;
		tmpidx[tmpentries-1]=1.0;
		tmpdist[tmpentries-1]=c;

		theSBLOCK.entries[i].dist=tmpdist; /* mem belongs to sblock now */
		theSBLOCK.entries[i].idx=tmpidx;
		theSBLOCK.entries[i].entry_enum=tmpentries;
		
		theSBLOCK.distance+=tmpdist[tmpentries-1];
		
	}

	/* 3) Iterate and plot based on p03_itercurve */
	if (theSBLOCK.distance>0) {
		interval=1.0/(theSBLOCK.distance*4.0);

		for (c=0; c<=1; c+=interval) {  /* loop over whole curve 0 to 1 */

			/* itercurve starts */

			distvar=c*theSBLOCK.distance;
			cumtot=0;

			for (ptlp=0; ptlp<theSBLOCK.entry_enum; ptlp++) {
				themin=theSBLOCK.entries[ptlp].dist[0];
				themax=theSBLOCK.entries[ptlp].dist[theSBLOCK.entries[ptlp].entry_enum-1];
				if (distvar>=themin+cumtot && distvar<=themax+cumtot) {
					minic=(distvar-cumtot)/(themax-themin);
					break;
				}
				cumtot+=themax;
			}


			minic*=theSBLOCK.entries[ptlp].dist[theSBLOCK.entries[ptlp].entry_enum-1];

			for (sfunc=0; sfunc<theSBLOCK.entries[ptlp].idx[theSBLOCK.entries[ptlp].entry_enum-1]; sfunc++) {
				if (minic>=theSBLOCK.entries[ptlp].dist[sfunc]) 
					minidx=sfunc;
			}
			maxidx=theSBLOCK.entries[ptlp].entry_enum-1;
			for (sfunc=0; sfunc<theSBLOCK.entries[ptlp].idx[theSBLOCK.entries[ptlp].entry_enum-1]; sfunc++) {
				if (minic<=theSBLOCK.entries[ptlp].dist[sfunc]) 
					maxidx=sfunc;
			}

			/* interpolate the s */
			if (maxidx==minidx) {
				/* its exact */
				s=theSBLOCK.entries[ptlp].idx[maxidx];
			}
			else {
				minval=theSBLOCK.entries[ptlp].dist[minidx];
				maxval=theSBLOCK.entries[ptlp].dist[maxidx];
				min_s=theSBLOCK.entries[ptlp].idx[minidx];
				max_s=theSBLOCK.entries[ptlp].idx[maxidx];
				ratio=((minic-minval)/(maxval-minval));
				s=((max_s-min_s)*ratio)+min_s;

			}

			g1x=*(GBLOCK+(ptlp*8));
			g2x=*((GBLOCK+(ptlp*8))+1);
			g3x=*((GBLOCK+(ptlp*8))+2);
			g4x=*((GBLOCK+(ptlp*8))+3);
			g1y=*((GBLOCK+(ptlp*8))+4);
			g2y=*((GBLOCK+(ptlp*8))+5);
			g3y=*((GBLOCK+(ptlp*8))+6);
			g4y=*((GBLOCK+(ptlp*8))+7);

			q1=(-0.5*s*s*s)+(1*s*s)+(-0.5*s)+0;
			q2=(1.5*s*s*s)+(-2.5*s*s)+(0*s)+1;
			q3=(-1.5*s*s*s)+(2*s*s)+(0.5*s)+0;
			q4=(0.5*s*s*s)+(-0.5*s*s)+(0*s)+0;

			Psx=g1x*q1 + g2x*q2 + g3x*q3 + g4x*q4;
			Psy=g1y*q1 + g2y*q2 + g3y*q3 + g4y*q4;

			
			/* plot a circle at Psy,Psx on canvas of radius 'thickness' */
			Psy=floor(Psy);
			Psx=floor(Psx);
			
			for (lpx=((int)floor(Psx))-maskoffset; lpx<=((int)floor(Psx))+maskoffset; lpx++) {
				for (lpy=((int)floor(Psy))-maskoffset; lpy<=((int)floor(Psy))+maskoffset; lpy++) {

					mskx=lpx-(((int)floor(Psx))-maskoffset);
					msky=lpy-(((int)floor(Psy))-maskoffset);

					if (lpx>=0 & lpx<width & lpy>=0 & lpy<height) {
							*(canvas+lpy+(lpx*height))+=*(maskblock+msky+(mskx*(maskoffset*2+1)));
					}
				}
			}
			
		}
	}

  /* Return the canvas to MATLAB */
	dims[0]=height;
  dims[1]=width;
  ansarray=mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
  ptr=mxGetData(ansarray);
  for (i=0; i<height*width; i++) {
		if (*(canvas+i)>0)
			*(ptr+i)=1;
		else
			*(ptr+i)=0;
  }

  dims[0]=1;
  dims[1]=1;
  plhs[0]=ansarray;
  ansarray=mxCreateNumericArray(2,dims,mxDOUBLE_CLASS,mxREAL);
  *mxGetPr(ansarray)=theSBLOCK.distance;
  plhs[1]=ansarray;

	for (i=0; i<theSBLOCK.entry_enum; i++) {
		mxFree(theSBLOCK.entries[i].idx);
		mxFree(theSBLOCK.entries[i].dist);
	}
	mxFree(theSBLOCK.entries);
	mxFree(GBLOCK);
	mxFree(G);
  mxFree(canvas);
	mxFree(pts);
	mxFree(maskblock);

}


int get_params(const mxArray* args[], int argc, int* height, int* width, double** pts, int* ptsenum, int* thickness) {

	const int* dims;
  int     i;

  if (argc!=4) {
        return FALSE;
  }

  /* Get height */
  if (!mxIsDouble(args[0]) || mxIsComplex(args[0]))
		return FALSE;	
	else {
		*height=(int)floor(*mxGetPr(args[0]));
	}

  /* Get width */
  if (!mxIsDouble(args[1]) || mxIsComplex(args[1]))
		return FALSE;	
	else {
		*width=(int)floor(*mxGetPr(args[1]));
	}

  /* Get thickness */
  if (!mxIsDouble(args[3]) || mxIsComplex(args[3]))
		return FALSE;	
	else {
		*thickness=(int)floor(*mxGetPr(args[3]));
	}

  /* Get control points */
  if (!mxIsDouble(args[2]) || mxIsComplex(args[2]))
		return FALSE;	
	else {
		dims=mxGetDimensions(args[2]);
		if (dims[0]!=2)
			return FALSE;
		if (dims[1]<4)
			return FALSE;
		*ptsenum=dims[1];
		*pts=(double*)mxCalloc((*ptsenum)*2,sizeof(double));

		for (i=0; i<2*(*ptsenum); i++) {
			*(*pts+i)=*(mxGetPr(args[2])+i);
		}
	}


	return TRUE;

}




