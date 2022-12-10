#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>  
#include <omp.h>
#include "mpi.h"

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
    MPI_File_read_at(fh,offset, numbers, count, MPI_DOUBLE, &status);
    int bytes_read;
    MPI_Get_count(&status, MPI_DOUBLE, &bytes_read);
    MPI_File_close(&fh);
    *p_size = count;
    return numbers;
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    int p_size_mat;
    int p_size_vec;
    double start = omp_get_wtime();
    double* matrix = par_read("/shared/Labs/Lab_2/matrix.bin", &p_size_mat, rank, size);
    double* vector = par_read("/shared/Labs/Lab_2/matrix_vector.bin", &p_size_vec, rank, size);
    
    //Before doing any operation, since each proceess might have several rows, we must know the number of rows the matrix has.
    int total_size =p_size_mat*size;
    int nrows = sqrt(total_size); //Because it is a square matrix
 
    int process_rows = nrows/size;
    double* result_loc = malloc(process_rows*nrows*sizeof(double));
    double* complete_vector = malloc(nrows*sizeof(double));
    
    //Every process needs the entire vector to perform the operations
    MPI_Allgather(vector, p_size_vec, MPI_DOUBLE, complete_vector, p_size_vec, MPI_DOUBLE, MPI_COMM_WORLD);

    //#pragma omp parallel for simd reduction(+: result_loc)
    for(int i = 0; i<process_rows;++i){
        result_loc[i] = 0;
        for(int j = 0; j<nrows;++j){
            result_loc[i] += matrix[(i*nrows+j)]*complete_vector[j];
        }
    }
    
    free(matrix);
    free(vector);
    free(complete_vector);

    MPI_Barrier(MPI_COMM_WORLD);

    double* final_result = malloc(sizeof(double)*nrows);
    MPI_Gather(result_loc, process_rows, MPI_DOUBLE, final_result, process_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(result_loc);
    double end =omp_get_wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank==0){
        //for(int i =0; i<nrows;++i)
        printf("Execution time = %f\nC[0]= %f\n", end-start, final_result[0]);
    } 
    MPI_Finalize();

}
