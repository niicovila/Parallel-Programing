#include "mpi.h"
#include <stdio.h>

int main( int argc, char *argv[] ){
    MPI_Init( &argc, &argv );
    
    int rank, size, len;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    if(rank==0) printf("\nPHASE 1\n\n");
    for(int i=0; i<size; ++i){
        if(rank==i) printf( "Hi, I am rank %d. My communicator is MPI_COMM_WORLD and has a size of %d processes\n", rank, size );
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    //PHASE 2
    MPI_Comm SPLIT;
    int color = (int)rank/4;
    int split_rank, split_size;
    char split_name[20];
    if(rank==0) printf("\nPHASE 2:\n\n");
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &SPLIT);
    MPI_Comm_rank(SPLIT , &split_rank);
    MPI_Comm_size(SPLIT, &split_size);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i<size;++i){
        if(rank==i){
            sprintf(split_name, "SPLIT_COMM_%d", color);
            MPI_Comm_set_name(SPLIT,split_name);
            printf("Hi, I was rank %d in communicator MPI_COMM_WORLD which had %d processes. Now I'm rank %d in communicator %s which has %d processes\n", rank, size, split_rank,split_name,split_size);
        }     
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // PHASE 3
    MPI_Barrier(MPI_COMM_WORLD);   
    if(rank==0) printf("\nPHASE 3\n\n");   
     
    MPI_Comm EVEN_COMM;
    MPI_Group eveng, splitg;
    int evens[] = {0,2,4,6,8,10,12,14};
    char even_name[20];
    int even_size, even_rank;
    sprintf(even_name, "EVEN_COMM");
    

    MPI_Comm_group(MPI_COMM_WORLD,&splitg);
    MPI_Group_incl(splitg,8,evens,&eveng);
    MPI_Comm_create(MPI_COMM_WORLD,eveng,&EVEN_COMM);
    if(EVEN_COMM != MPI_COMM_NULL){
         MPI_Comm_set_name(EVEN_COMM,even_name);
         MPI_Comm_size(EVEN_COMM,&even_size);
         MPI_Comm_rank(EVEN_COMM,&even_rank);
         for(int i = 0; i < even_size; i++){
              if(rank == evens[i]) printf("Hi, I was rank %d in communicator %s which had %d processes. Now I'm rank %d in communicator %s which has %d processes.\n", split_rank,split_name ,split_size, even_rank, even_name, even_size );
               fflush(stdout);
        }
    }
    MPI_Group_free(&eveng);
    //MPI_Group_free(&splitg);

    // PHASE 4
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("\nPHASE 4\n\n");
    
    MPI_Comm ODD_COMM;
    MPI_Group oddg;
    char odd_name[20];
    int odd_size, odd_rank;
    sprintf(odd_name, "ODD_COMM");
    
    MPI_Group_excl(splitg,8,evens,&oddg);
    MPI_Comm_create(MPI_COMM_WORLD,oddg,&ODD_COMM);
    if(ODD_COMM != MPI_COMM_NULL){

        MPI_Comm_set_name(ODD_COMM, odd_name);
        MPI_Comm_rank(ODD_COMM,&odd_rank);
        MPI_Comm_size(ODD_COMM, &odd_size);
        for(int i = 0;i < odd_size; i++){
	    if(odd_rank == i) printf("Hi, I was rank %d in communicator MPI_COMM_WORLD which had %d processes. Now I'm rank %d in communicator %s which has %d processes.\n",rank,size, odd_rank, odd_name, odd_size);
 	    fflush(stdout);
        }
    }

    MPI_Group_free(&oddg);
    
    MPI_Group_free(&splitg);
    MPI_Finalize();
    return 0;
}
