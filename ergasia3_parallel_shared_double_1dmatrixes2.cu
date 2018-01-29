#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <time.h>
#include <cuda.h>

struct timeval startwtime,endwtime;
double seq_time;

__device__ double kernel(double a,double *d_sigma);
__device__ double distance(int l,int q,double *a, double *b, int *d_d);
void printmatrixd(int a, int b, double *matrix);
void printmatrixi(int a,int b, int *matrix);
void printsolution(int a, int b, double *point, int *kmatrix);
void checkfunction(void);

__global__ void devMain(double *d_m,int *d_kmatrix,double *d_x,double *d_y,double *d_y2, double *d_epsilon, double *d_sigma, int *d_n, int *d_d, int *d_count, double *d_xtemp, double *d_sum2, int *d_numthreads, double *d_sum);

int i,j,s,n,d,len,h,numthreads;
int *d_n, *d_d;

double *sum2,*m, *xtemp, *sum;
double *d_sum2, *d_m, *d_xtemp, *d_sum;

int *kmatrix, *count;
int *d_kmatrix, *d_count, *d_numthreads;

FILE *myfile;

double *x, *y, *y2;
double *d_x, *d_y, *d_y2;

double sigma,epsilon,p; //only use sigma and epsilon
double *d_sigma, *d_epsilon;

int main(int argc, char **argv){
	numthreads=6 ;// you can change this if you want
	if (argc!=5) {
		printf("Usage: %s, s, n, d, file.bin, \nwhere s is sigma*0.1, e is sigma*0.0001, \nn is the number of elements, \nd its dimension and file.bin\nthe binary file with the elements \n",argv[0]);
	    exit(1);
	}
	s=atoi(argv[1]);
	sigma=s*0.1;
	cudaMalloc((void **)&d_sigma,sizeof(double));
	cudaMemcpy(d_sigma,&sigma,sizeof(double),cudaMemcpyHostToDevice);

	
	cudaMalloc((void **)&d_numthreads,sizeof(int));
	cudaMemcpy(d_numthreads,&numthreads,sizeof(int),cudaMemcpyHostToDevice);



	epsilon=sigma*0.0001;
	cudaMalloc((void **)&d_epsilon,sizeof(double));
	cudaMemcpy(d_epsilon,&epsilon,sizeof(double),cudaMemcpyHostToDevice);

	n=atoi(argv[2]);
	cudaMalloc((void **)&d_n,sizeof(int));
	cudaMemcpy(d_n,&n,sizeof(int),cudaMemcpyHostToDevice);

	d=atoi(argv[3]);
	cudaMalloc((void **)&d_d,sizeof(int));
	cudaMemcpy(d_d,&d,sizeof(int),cudaMemcpyHostToDevice);

	xtemp=(double*)malloc(n*numthreads*sizeof(double));
	cudaMalloc((void **)&d_xtemp,n*numthreads*sizeof(double));

	m=(double*)malloc(n*sizeof(double));
	cudaMalloc((void **)&d_m,n*sizeof(double));

	sum2=(double*)malloc(n*numthreads*sizeof(double));
	cudaMalloc((void **)&d_sum2,n*numthreads*sizeof(double));

	sum=(double*)malloc(n*sizeof(double));
	cudaMalloc((void **)&d_sum,n*sizeof(double));

	kmatrix=(int*)malloc(n*sizeof(int));
	cudaMalloc((void **)&d_kmatrix,n*sizeof(int));

	count=(int*)malloc(n*sizeof(int));
	cudaMalloc((void **)&d_count,n*sizeof(int));

	x=(double*)malloc(n*d*sizeof(double));
	cudaMalloc((void **)&d_x,n*d*sizeof(double));

	y=(double*)malloc(n*d*sizeof(double));
	cudaMalloc((void **)&d_y,n*d*sizeof(double));

	y2=(double*)malloc(n*d*sizeof(double));
	cudaMalloc((void **)&d_y2,n*d*sizeof(double));

	for (i=0;i<n;i++){
		m[i]=100;
		for (j=0;j<numthreads;j++){
			xtemp[i*numthreads+j]=0;
			sum2[i*numthreads+j]=0;
		}
		sum[i]=0;
		kmatrix[i]=0;
		count[i]=0;
	}
	cudaMemcpy(d_m,m,n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum,sum,n*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum2,sum2,n*numthreads*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_kmatrix,kmatrix,n*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_xtemp,xtemp,n*numthreads*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_count,count,n*sizeof(int),cudaMemcpyHostToDevice);

	myfile=fopen(argv[4],"rb");
	for (i=0;i<n;i++){
		for (j=0;j<d;j++){
			len=fread(&p,8,1,myfile);
			x[i*d+j]=p;
			y[i*d+j]=0;
			y2[i*d+j]=p;
		}
	}
	fclose(myfile);

	cudaMemcpy(d_x,x,n*d*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,n*d*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2,y2,n*d*sizeof(double),cudaMemcpyHostToDevice);
	/** start of the main part of the program**/

	gettimeofday(&startwtime,NULL);

	devMain<<<n,numthreads,n*d*sizeof(double)>>>(d_m,d_kmatrix,d_x,d_y,d_y2,d_epsilon,d_sigma,d_n,d_d,d_count,d_xtemp,d_sum2,d_numthreads,d_sum);

	gettimeofday(&endwtime,NULL);
	cudaMemcpy(kmatrix,d_kmatrix,n*sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(y2,d_y2,n*d*sizeof(double),cudaMemcpyDeviceToHost);

	printsolution(n,d,y2,kmatrix);
	//checkfunction();

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
	printf("total time: %f secs\n",seq_time);

    free(x);
    free(y);
    free(y2);
    free(count);
    free(kmatrix);
    free (sum);
    free (sum2);
    free(m);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y2);
    cudaFree(d_count);
    cudaFree(d_kmatrix);
    cudaFree(d_sum2);
    cudaFree(d_sum);
    cudaFree(m);

}

__device__ double kernel(double a, double *d_sigma){
	double result;
	result=exp(-a/(2*(*d_sigma)*(*d_sigma)));
	return result;
}

__device__ double distance(int l,int q, double *a, double *b, int *d_d){
	double sum=0;
	double root=0;
	int j;
	for (j=0;j<(*d_d);j++){
		sum+=(a[l*(*d_d)+j]-b[q*(*d_d)+j])*(a[l*(*d_d)+j]-b[q*(*d_d)+j]);
	}
	root=sqrt(sum);
	return root;
}

void printmatrixd(int a,int b, double *matrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%f ",matrix[i*b+j]);
		}
		printf("\n");
	}
}

void printmatrixi(int a,int b, int *matrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%d ",matrix[i*b+j]);
		}
		printf("\n");
	}
}

void printsolution(int a, int b, double *point, int *kmatrix){
	int i,j;
	for (i=0;i<a;i++){
		for (j=0;j<b;j++){
			printf("%f ",point[i*b+j]);
		}
		printf("i=%d, k=%d\n",i, kmatrix[i]);
	}
}

void checkfunction(void){
	char binary[50];
	FILE *myfile2;
	int  count,i,j;
	double *a;
	double p,dist;
	a=(double*)malloc(n*d*sizeof(double));
	printf("please give the name of the binary file \n(of double floats-8-bytes) that is a right solution\nto the mean shift problem:\n");
	dist=0;
	count=0;

	scanf("%s",binary);
	myfile2=fopen(binary,"rb");
	for (i=0;i<n;i++){
		//printf("reached here\n");
		for (int j=0;j<d;j++){
			fread(&p,8,1,myfile);
			a[i*d+j]=p;
		}
	}
	fclose(myfile2);
	for (i=0;i<n;i++){
		for (j=0;j<d;j++){
			dist+=(a[i*d+j]-y2[i*d+j])*(a[i*d+j]-y2[i*d+j]);
		}
		if (dist>sigma*sigma/100){ //distance>sigma/10
			count++;
		}
		dist=0;
	}
	free(a);
	printf("we have problem in %d out of %d points\n",count,n);



}

__global__ void devMain(double *d_m,int *d_kmatrix,double *d_x,double *d_y,double *d_y2, double *d_epsilon, double *d_sigma, int *d_n, int *d_d, int *d_count, double *d_xtemp, double *d_sum2, int *d_numthreads, double *d_sum){
	int j,i;
	int h;
	int bid=blockIdx.x;
	int tid=threadIdx.x;
	int steps=(*d_n/(*d_numthreads));
	extern __shared__ double sharedx[];
	

	for (j=0;j<(*d_d);j++){
		sharedx[bid*(*d_d)+j]=d_x[bid*(*d_d)+j];
	}
	__syncthreads();

		while ((d_m[bid]>*d_epsilon)&&(d_kmatrix[bid]<15)){
			//d_count[bid]=0;
			for (j=tid*steps;j<(tid+1)*steps;j++){
				//printf("j=%d\n",j);
				if (distance(bid,j,d_y2,d_x,d_d)<(*d_sigma)*(*d_sigma)){
					d_xtemp[bid*(*d_numthreads)+tid]=kernel(distance(bid,j,d_y2,d_x,d_d),d_sigma);
					//d_count[bid*(*d_numthreads)+tid]++;
					d_sum2[bid*(*d_numthreads)+tid]+=d_xtemp[bid*(*d_numthreads)+tid];
					for (h=0;h<(*d_d);h++){
						for (i=0;i<(*d_numthreads);i++){
						if (i==tid)	//to prevent data races
						d_y[bid*(*d_d)+h]+=d_xtemp[bid*(*d_numthreads)+tid]*d_x[j*(*d_d)+h];
					    __syncthreads();
					}
					}
				}
			}
			if (1==tid){
			for (h=0;h<(*d_numthreads);h++){
				d_sum[bid]+=d_sum2[bid*(*d_numthreads)+h];
			}
			for (h=0;h<*d_d;h++){
				d_y[bid*(*d_d)+h]/=d_sum[bid];
			}
			d_sum[bid]=0;
			d_m[bid]=distance(bid,bid,d_y,d_y2,d_d);
			d_kmatrix[bid]++;
			for (h=0;h<(*d_d);h++){
				d_y2[bid*(*d_d)+h]=d_y[bid*(*d_d)+h];
				d_y[bid*(*d_d)+h]=0;
			}
		}
		__syncthreads();
		d_sum2[bid*(*d_numthreads)+tid]=0;
		__syncthreads();
	}
	
}