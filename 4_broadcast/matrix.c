#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>  
#include <omp.h>
#include "mpi.h"

int *new_matrix(int size, int rank){
    int* matrix = malloc(sizeof(int)*size*size);
    for(int i = 0; i<size;++i){
        for(int j = 0; j<size;++j){
            if(i == j){
                matrix[i*size+j] = rank;
            }else{
                matrix[i*size+j] = 0;
            }
        }
    }
    return matrix;
}
void print_matrix(int* matrix, int size){
    for(int i = 0; i<size;++i){
        for(int j = 0; j<size;++j){
            printf("%d ", matrix[(i*size+j)]);
            if(j==size-1) printf("\n");
        }
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    //MPI_Status status;
    //double start = omp_get_wtime();
    int size;
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    int* matrix = new_matrix(size, rank);
    if(rank==0){
       printf("Initial Matrix (rank 0)\n");
       print_matrix(matrix, size);
       printf("\n\n");
    }
    MPI_Datatype diagonal_type;
    MPI_Type_vector(size, 1, size+1, MPI_INT, &diagonal_type);
    MPI_Type_commit(&diagonal_type);
    
    MPI_Gather(&matrix[0], 1, diagonal_type, matrix, size, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Send(&matrix[0], 1, diagonal_type, 0, rank, MPI_COMM_WORLD);
    MPI_Type_free(&diagonal_type);


    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
        printf("Final Matrix (rank 0)\n");
        print_matrix(matrix, size);
    }
    //double end = omp_get_wtime();
    //printf("Total time:%f\n", end-start);
    free(matrix);
    MPI_Finalize();

}
