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

   if(indy >= 0 && indy < ny){
       if(indx >= 0 && indx < nx){
           image_invert[pixel(indx,indy,nx)] = 255 - image[pixel(indx,indy,nx)];
       }
   }

}

/*smooth*/
__global__ void smooth(int* image, int* image_smooth, int nx, int ny){

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indy >= 0 && indy < ny){
       if(indx >= 0 && indx < nx){
           if ( indx == 0 || indy == 0 || indx == nx-1 || indy == ny-1){
               image_smooth[pixel(indx,indy,nx)] = 0;
           } else {
               image_smooth[pixel(indx,indy,nx)] = (1.0 / 9.0) * (image[pixel(indx - 1, indy + 1, nx)] + image[pixel(indx, indy + 1, nx)] +
                                                         image[pixel(indx + 1, indy + 1, nx)] + image[pixel(indx - 1, indy, nx)] +
                                                         image[pixel(indx, indy, nx)] + image[pixel(indx + 1, indy, nx)] +
                                                         image[pixel(indx - 1, indy - 1, nx)] + image[pixel(indx, indy - 1, nx)] +
                                                         image[pixel(indx + 1, indy - 1, nx)]);
           }
       }
   }
}

/*detect*/
__global__ void detect(int* image, int* image_detect, int nx, int ny) {

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indy >= 0 && indy < ny){
       if(indx >= 0 && indx < nx){
           if (indx == 0 || indy == 0 || indx == nx - 1 || indy == ny - 1) {
                image_detect[pixel(indx, indy, nx)] = 0;
            } else {
                image_detect[pixel(indx, indy, nx)] =
                        image[pixel(indx - 1, indy, nx)] + image[pixel(indx + 1, indy, nx)] + image[pixel(indx, indy - 1, nx)] +
                        image[pixel(indx, indy + 1, nx)] - 4.0 * image[pixel(indx, indy, nx)];
                image_detect[pixel(indx,indy,nx)] = fmin(image_detect[pixel(indx,indy,nx)], 255);
                image_detect[pixel(indx,indy,nx)] = fmax(image_detect[pixel(indx,indy,nx)], 0);
            }
       }
   }
}

/*enhance*/
__global__ void enhance(int* image,int *image_enhance,int nx, int ny){

   int indx = threadIdx.x + blockIdx.x * blockDim.x;
   int indy = threadIdx.y + blockIdx.y * blockDim.y;

   if(indy >= 0 && indy < ny){
       if(indx >= 0 && indx < nx){
           if (indx == 0 || indy == 0 || indx== nx - 1 || indy == ny - 1) {
                image_enhance[pixel(indx, indy, nx)] = 0;
            }else {
                image_enhance[pixel(indx, indy, nx)] = 5.0 * image[pixel(indx, indy, nx)] -
                                                 (image[pixel(indx - 1, indy, nx)] + image[pixel(indx + 1, indy, nx)] +
                                                  image[pixel(indx, indy - 1, nx)] + image[pixel(indx, indy + 1, nx)]);
                image_enhance[pixel(indx,indy,nx)] = fmin(image_enhance[pixel(indx,indy,nx)],255);
                image_enhance[pixel(indx,indy,nx)] = fmax(image_enhance[pixel(indx,indy,nx)],0);
	        }
       }
   }
}

void allocate(int** h_image, int** h_image_invert, int** h_image_smooth, 
              int** h_image_detect,int** h_image_enhance, int** d_image, int** d_image_invert,
              int** d_image_smooth, int** d_image_detect, int** d_image_enhance, int nx, int ny ){

    *h_image=(int *) malloc(sizeof(int)*nx*ny); 
    *h_image_invert  = (int *) malloc(sizeof(int)*nx*ny);  
    *h_image_smooth  = (int *) malloc(sizeof(int)*nx*ny);  
    *h_image_detect  = (int *) malloc(sizeof(int)*nx*ny);  
    *h_image_enhance = (int *) malloc(sizeof(int)*nx*ny); 

    cudaMalloc((void**)d_image, nx*ny*sizeof(int));
    cudaMalloc((void**)d_image_invert, nx*ny*sizeof(int));
    cudaMalloc((void**)d_image_smooth, nx*ny*sizeof(int));
    cudaMalloc((void**)d_image_detect, nx*ny*sizeof(int));
    cudaMalloc((void**)d_image_enhance, nx*ny*sizeof(int));
}
void deallocate(int** h_image, int** h_image_invert, int** h_image_smooth, 
                int** h_image_detect, int** h_image_enhance, int** d_image, int** d_image_invert,
                int** d_image_smooth, int** d_image_detect, int** d_image_enhance, int nx, int ny ){

    free(*h_image);
    free(*h_image_invert);
    free(*h_image_smooth);
    free(*h_image_detect);
    free(*h_image_enhance);

    cudaFree(*d_image);
    cudaFree(*d_image_invert);
    cudaFree(*d_image_smooth);
    cudaFree(*d_image_detect);
    cudaFree(*d_image_enhance);
}

/* Main program */
int main (int argc, char *argv[])
{
   int    nx,ny;
   int B = 16;
   int *h_image, *h_image_invert, *h_image_smooth, *h_image_detect, *h_image_enhance;
   int *d_image, *d_image_invert, *d_image_smooth, *d_image_detect, *d_image_enhance;
   char   filename[250];
  // double runtime=0;

   /* Get parameters */
   if (argc != 4) 
   {
      printf ("Usage: %s image_name N M \n", argv[0]);
      exit (1);
   }
   sprintf(filename, "%s.txt", argv[1]);
   nx  = atoi(argv[2]);
   ny  = atoi(argv[3]);

   printf("%s %d %d\n", filename, nx, ny);
   
   dim3 dimBlock(B,B,1);
   int dimgx = (nx+B-1)/B;
   int dimgy = (ny+B-1)/B;
   dim3 dimGrid(dimgx, dimgy,1);


   /* Allocate pointers */
   allocate(&h_image, &h_image_invert, &h_image_smooth, &h_image_detect, &h_image_enhance, 
            &d_image, &d_image_invert, &d_image_smooth, &d_image_detect, &d_image_enhance, nx, ny );

   /* Read image and save in array imgage */
   readimg(filename,nx,ny,h_image);

   //time_t start = time(NULL);
   float runtime;
   cudaEvent_t start,stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start);

   cudaMemcpy(d_image,h_image,nx*ny*sizeof(int),cudaMemcpyHostToDevice);

   invert<<<dimGrid,dimBlock,sizeof(double)*B*B>>>(d_image, d_image_invert, nx, ny);
   cudaMemcpy(h_image_invert,d_image_invert,nx*ny*sizeof(int),cudaMemcpyDeviceToHost);

   smooth<<<dimGrid,dimBlock,sizeof(double)*B*B>>>(d_image, d_image_smooth, nx, ny);
   cudaMemcpy(h_image_smooth,d_image_smooth,nx*ny*sizeof(int),cudaMemcpyDeviceToHost);

   detect<<<dimGrid,dimBlock,sizeof(double)*B*B>>>(d_image, d_image_detect, nx, ny);
   cudaMemcpy(h_image_detect,d_image_detect,nx*ny*sizeof(int),cudaMemcpyDeviceToHost);

   enhance<<<dimGrid,dimBlock,sizeof(double)*B*B>>>(d_image, d_image_enhance, nx, ny);
   cudaMemcpy(h_image_enhance,d_image_enhance,nx*ny*sizeof(int),cudaMemcpyDeviceToHost);
   
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&runtime,start, stop);
   
  // runtime = (double)(time(NULL)- start);
   printf("Total time: %f\n",runtime);

   /* Save images */
   char fileout[255]={0};
   sprintf(fileout, "%s-inverse.txt", argv[1]);
   saveimg(fileout,nx,ny,h_image_invert);
   sprintf(fileout, "%s-smooth.txt", argv[1]);
   saveimg(fileout,nx,ny,h_image_smooth);
   sprintf(fileout, "%s-detect.txt", argv[1]);
   saveimg(fileout,nx,ny,h_image_detect);
   sprintf(fileout, "%s-enhance.txt", argv[1]);
   saveimg(fileout,nx,ny,h_image_enhance);

   /* Deallocate  */
   deallocate(&h_image, &h_image_invert, &h_image_smooth, &h_image_detect, &h_image_enhance, 
              &d_image, &d_image_invert, &d_image_smooth, &d_image_detect, &d_image_enhance, nx, ny );

}
