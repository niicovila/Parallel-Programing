#include "mpi.h"
#include "omp.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double *par_read(char *in_file, int *p_size, int rank, int nprocs){
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, in_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset size;
    MPI_File_get_size(fh, &size);
    int process_size = (size/nprocs);
    double* numbers = malloc(process_size);
    MPI_Offset offset = rank*process_size;
    int count = process_size/sizeof(double);
    MPI_File_read_at(fh, offset, numbers, count, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    *p_size = count;
    return numbers;
}


int main(int argc, char* argv[]){
    double global_dotp;
    MPI_Init(&argc, &argv);
    double result_loc;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    int p_size;
    double start = omp_get_wtime();
    double* arr1 = par_read("/shared/Labs/Lab_2/array_p.bin", &p_size, rank, size);
    double* arr2 = par_read("/shared/Labs/Lab_2/array_q.bin", &p_size, rank, size);
 
    #pragma omp parallel for simd reduction(+: result_loc)
    for(int i = 0; i<p_size; ++i){
	    result_loc += arr1[i]*arr2[i];
    
    }
    //printf("result of rank %d is: %f\n", rank, result_loc); 
    free(arr1);
    free(arr2);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&result_loc, &global_dotp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double end = omp_get_wtime();
    double runtime = end-start;
    if(rank==0) printf("Dot ptoduct result: %f\nTotal time: %f\n", global_dotp, runtime);
    MPI_Finalize();
}
