#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <time.h>

struct timeval startwtime,endwtime;
double seq_time;

double kernel(double a);
double distance(int l,int q,double **a, double **b);
void printmatrixd(int a, int b, double **matrix);
void printmatrixi(int a,int b, int **matrix);
void printsolution(int a, int b, double **point, int *kmatrix);

int i,j,s,n,d,len,h;
double *sum2,*m, *xtemp;
int *kmatrix, *count;
FILE *myfile;
double **x, **y, **y2;
double sigma,epsilon,p; //only use sigma and epsilon

int main(int argc, char **argv){
	if (argc!=5) {
		printf("Usage: %s, s, n, d, file.bin, \nwhere s is sigma*0.1, e is sigma*0.0001, \nn is the number of elements, \nd its dimension and file.bin\nthe binary file with the elements \n",argv[0]);
	    exit(1);
	}
	s=atoi(argv[1]);
	sigma=s*0.1;
	epsilon=sigma*0.0001;
	n=atoi(argv[2]);
	d=atoi(argv[3]);

	xtemp=(double*)malloc(n*sizeof(double));
	m=(double*)malloc(n*sizeof(double));
	sum2=(double*)malloc(n*sizeof(double));
	kmatrix=(int*)malloc(n*sizeof(int));
	count=(int*)malloc(n*sizeof(int));

	y2=malloc(n*sizeof *y2);
	y=malloc(n*sizeof *y);
	x=malloc(n*sizeof *x);
	for (i=0;i<n;i++){
		x[i]=malloc(d*sizeof *x[i]);
		y[i]=malloc(d*sizeof *y[i]);
		y2[i]=malloc(d*sizeof *y2[i]);
		m[i]=10;
		sum2[i]=0;
		kmatrix[i]=0;
		xtemp[i]=0;
		count[i]=0;
	}


	myfile=fopen(argv[4],"rb");
	for (i=0;i<n;i++){
		for (j=0;j<d;j++){
			len=fread(&p,8,1,myfile);
			x[i][j]=p;
			y[i][j]=0;
			y2[i][j]=p;
		}
	}
	fclose(myfile);
	/** start of the main part of the program**/

	gettimeofday(&startwtime,NULL);

	for (i=0;i<n;i++){    //i<n
		while ((m[i]>epsilon)&&(kmatrix[i]<15)){
			count[i]=0;
			for (j=0;j<n;j++){
				//printf("j=%d\n",j);
				if (distance(i,j,y2,x)<sigma*sigma){
					xtemp[i]=kernel(distance(i,j,y2,x));
					count[i]++;
					sum2[i]+=xtemp[i];
					for (h=0;h<d;h++){
						y[i][h]+=xtemp[i]*x[j][h];
					}
				}
			}
			for (h=0;h<d;h++){
				y[i][h]/=sum2[i];
			}
			sum2[i]=0;
			m[i]=distance(i,i,y,y2);
			kmatrix[i]++;
			/**printf("m[i]=%f ",m[i]);
			printf("%f ",y[i][0]);
			printf("%f ",y[i][1]);
			printf("i=%d ",i);
			printf("k=%d ",kmatrix[i]);
			printf("count=%d\n",count[i]);
			//printf("d=%f, i=%d\n",distance(i,i,y,y3),i);
			//printf("i=%d\n",i);**/
			for (h=0;h<d;h++){
				y2[i][h]=y[i][h];
				y[i][h]=0;
			}
		}
	}

	gettimeofday(&endwtime,NULL);

	printsolution(n,d,y2,kmatrix);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	printf("total time: %f secs\n",seq_time);

    for (i=0;i<n;i++){
    	free(x[i]);
    	free(y[i]);
    	free(y2[i]);
    }
    free(x);
    free(y);
    free(y2);  
}

double kernel(double a){
	double result;
	result=exp(-a/(2*sigma*sigma));
	return result;
}

double distance(int l,int q, double **a, double **b){
	double sum=0;
	double root=0;
	int j;
	for (j=0;j<d;j++){
		sum+=(a[l][j]-b[q][j])*(a[l][j]-b[q][j]);
	}
	root=sqrt(sum);
	return root;
}

void printmatrixd(int a,int b, double **matrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%f ",matrix[i][j]);
		}
		printf("\n");
	}
}

void printmatrixi(int a,int b, int **matrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%d ",matrix[i][j]);
		}
		printf("\n");
	}
}

void printsolution(int a, int b, double **point, int *kmatrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%f ",point[i][j]);
		}
		printf("i=%d, k=%d\n",i, kmatrix[i]);
	}
}
