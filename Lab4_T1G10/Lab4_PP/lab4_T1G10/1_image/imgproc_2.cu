/*
 *     
 *  IMAGE PROCESSING
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"
#define pixel(i, j, n)  (((j)*(n)) +(i))


/*read*/
void  readimg(char * filename,int nx, int ny, int * image){
  
   FILE *fp=NULL;

   fp = fopen(filename,"r");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fscanf(fp,"%d", &image[pixel(i,j,nx)]);      
      }
   }
   fclose(fp);
}

/* save */   
void saveimg(char *filename,int nx,int ny,int *image){

   FILE *fp=NULL;
   fp = fopen(filename,"w");
   for(int j=0; j<ny; ++j){
      for(int i=0; i<nx; ++i){
         fprintf(fp,"%d ", image[pixel(i,j,nx)]);      
      }
      fprintf(fp,"\n");
   }
   fclose(fp);

}

/*invert*/
__global__ void invert(int* image, int* image_invert, int nx, int ny){
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    printf("%d\n",nx);
    if (indx > 0 && indx < nx-1){
      if(indy > 0 && indy < ny-1){
         image_invert[pixel(indx, indy, nx)] = 255-image[pixel(indx, indy, nx)];
      }
    }
}

/*smooth*/
__global__ void smooth(int* image, int* image_smooth, int nx, int ny){

}

/*detect*/
__global__ void detect(int* image, int* image_detect, int nx, int ny){
   
}

/*enhance*/
__global__ void enhance(int* image,int *image_enhance,int nx, int ny){
   

}

/* Main program */
int main (int argc, char *argv[])
{
   int    nx,ny;
   char   filename[250];

   /* Get parameters */
   if (argc != 4) 
   {
      printf ("Usage: %s image_name N M \n", argv[0]);
      exit (1);
   }
   sprintf(filename, "%s.txt", argv[1]);
   nx  = atoi(argv[2]);
   ny  = atoi(argv[3]);
   int *d_nx, *d_ny;
   printf("%s %d %d\n", filename, nx, ny);
   float runtime;
   /* Allocate CPU and GPU pointers */

   int*   image=(int *) malloc(sizeof(int)*nx*ny); 
   int*   image_invert  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_smooth  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_detect  = (int *) malloc(sizeof(int)*nx*ny);  
   int*   image_enhance = (int *) malloc(sizeof(int)*nx*ny); 
   
   printf("a\n");

   cudaMalloc((void **)&d_nx, sizeof(int));
   cudaMalloc((void **)&d_ny, sizeof(int));
   printf("b\n");
   cudaMemcpy(d_nx, &nx, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ny, &ny, sizeof(int), cudaMemcpyHostToDevice);
   
   int *d_image_invert, *d_image;
   cudaMalloc((void **)&d_image_invert, nx*ny*sizeof(int));
   cudaMalloc((void **)&d_image, nx*ny*sizeof(int));
   printf("\c");
   cudaEvent_t start, stop; 
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   /* Read image and save in array imgage */
   readimg(filename,nx,ny,image);

   cudaMemcpy(d_image, image, nx*ny*sizeof(int), cudaMemcpyHostToDevice);
   int B=16;
   dim3 dimBlock(B, B, 1);
   int dimgx = (nx+B-1)/B;
   int dimgy = (ny+B-1)/B;
   dim3 dimGrid(dimgx, dimgy,1);
   printf("d\n");
   invert<<<dimBlock,dimGrid>>>(d_image,d_image_invert, *d_nx, *d_ny);

   cudaMemcpy(image_invert, d_image_invert, nx*ny*sizeof(int), cudaMemcpyDeviceToHost);
   cudaFree(d_image_invert); //cudaFree(d_nx); cudaFree(d_ny); cudaFree(d_image);

   /* Print runtime */
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&runtime, start, stop);
   printf("Total runtime is &f\n", runtime);
   
   /* Save images */
   char fileout[255]={0};
   sprintf(fileout, "%s-inverse.txt", argv[1]);
   saveimg(fileout,nx,ny,image_invert);
   sprintf(fileout, "%s-smooth.txt", argv[1]);
   saveimg(fileout,nx,ny,image_smooth);
   sprintf(fileout, "%s-detect.txt", argv[1]);
   saveimg(fileout,nx,ny,image_detect);
   sprintf(fileout, "%s-enhance.txt", argv[1]);
   saveimg(fileout,nx,ny,image_enhance);

   /* Deallocate CPU and GPU pointers*/
   free(image);
   free(image_invert);
   free(image_smooth);
   free(image_detect);
   free(image_enhance);
}
