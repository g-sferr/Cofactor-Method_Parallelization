
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_functions.h"						// sincronizzazione
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <time.h> 
#include <math.h>


#define N 50
#define cBlockdim 256

/*
 * _Protos_
 */


 /********** COFACTOR METHOD : CPU SIDE**********/

void // Function to calculate and store inverse unsing La Place; returns false if matrix is singular
InverseCofactor(float** Matrix, int dim, float detValue, float** inverseStore);

void // Function to calculate and store inverse using Gauss; returns false if matrix is singular
InverseCofactor2(float** Matrix, int dim, float detValue, float** inverseStore);

/********** FUNCTIONS UTILITIES **********/

void //Filling of Matrix
fillinMatrix(float** Matrix);

void //Print of Matrix
printMatrix(float** Matrix, int n, int m);

void
printMatrix2(float* Matrix, int n);

float//Determinant La Place
detMat(float** Matrix, int dim);

float//Determinant Gauss
detMatGauss(float** Matrix, int dim);


/********** GPU Side **********/

// Version using La Place
__device__
void detMat_GPU(float* Matrix, int dim, float* det);

__global__ //USABLE ONLY with bidimensional configuration of grid and block
void InverseCofactor_GPU(float* Matrix, int* dim, float* detValue, float* inverseStore);

// Version using Gauss (Kernel BASE) 
__global__ //USABLE ONLY with bidimensional configuration of grid and block
void InverseCofactor_DetGauss_GPU(float* Matrix, int* dim, float* detValue, float* inverseStore);

__global__ //USABLE ONLY with unidimensional configuration of grid and block
void InverseCofactor_DetGauss_GPU_minor(float* Matrix, int* dim, float* detValue, float* inverseStore);

// Version using Gauss (Kernel Optimized)
__global__ //USABLE ONLY with bidimensional configuration of grid and block
void InverseCofactor_DetGauss_GPU_2(float* Matrix, int* dim, float* detValue, float* inverseStore);

__global__ //USABLE ONLY with unidimensional configuration of grid and block
void InverseCofactor_DetGauss_GPU_2_minor(float* Matrix, int* dim, float* detValue, float* inverseStore);

/* Version using Gauss(Kernel BASE shared)
__global__ //USABLE ONLY with unidimensional configuration of grid and block
void InverseCofactor_DetGauss_GPU_minorshared(float* Matrix, int* dim, float* detValue, float* inverseStore);*/


int main()	/* ------------------------------ main() ------------------------------ */
{
	/* --- DECLARATIONS --- */

	/**** Analysis files creation File Management 
	FILE* pf_value;
	FILE* pf_text;

	pf_value = fopen("test_value_N50minor.txt", "a");
	if (pf_value == NULL)
	{
		printf("Impossibile aprire file test_value_N50minor!\n");
		exit(1);
	}

	pf_text = fopen("test_text_N50minor.txt", "a");
	if (pf_text == NULL)
	{
		printf("Impossibile aprire file test_text_N50minor!\n");
		exit(1);
	}
	****/ //Uncomment to use file generation in testing phase

	/****  ****/
	float** Matrix = (float**)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N; i++)
	{
		Matrix[i] = (float*)malloc(sizeof(float) * N);
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Matrix[i][j] = 0;
		}
	}

	float** inverseMat = (float**)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N; i++)
	{
		inverseMat[i] = (float*)malloc(sizeof(float) * N);
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			inverseMat[i][j] = 0;
		}
	}

	/****  ****/
	int size_m = N * N * sizeof(float);
	int dim = N; // copia host dim = N

	float* h_inversemat;
	h_inversemat = (float*)malloc(size_m);

	float* h_matrix; //copia host matrix passed as array
	h_matrix = (float*)malloc(size_m);

	float* h_inversemat2; //copia host ritorno kernel version 2
	h_inversemat2 = (float*)malloc(size_m);

	float* h_inversemat3; //copia host ritorno kernel version 2
	h_inversemat3 = (float*)malloc(size_m);

	/****  ****/
	float* d_matrix;
	int* d_dim;
	float* d_detValue;
	float* d_inverseStore;

	float* d_inverseStore2; //return of kernel version 2

	float* d_inverseStore3; //return of kernel version 3

	/****  ****/
	clock_t before;
	clock_t after;
	clock_t difference;
	int msec = 0, s = 0;
	float msecF;

	/****  ****/
	cudaError_t cudaStatus;

	cudaEvent_t start, stop;
	float elapsed = 0;
	int elapsedI;

	cudaEvent_t start2, stop2;
	float elapsed2 = 0;
	int elapsedI2;

	cudaEvent_t start3, stop3;
	float elapsed3 = 0;
	int elapsedI3;

	/****  ****/
	int grid_n = 10;
	int block_n = cBlockdim;

	dim3 block(N, N);
	dim3 grid(1, 1);


	//for (int t = 0; t < 50 ; t++){ //begin test for (50 iteration)
		//printf("iter %d\n",t);


		/* -- Matrix Building -- */

	fillinMatrix(Matrix); //Random Filling of Matrix

	printf("\n\n\t=== INPUT Matrix ===\n");
	//printMatrix(Matrix, N, N);

	/* -- Check for Non-singularity of the matrix -- */
	float detValue = detMatGauss(Matrix, N);
	printf("\n>> Determinant = %.4f\n", detValue);
	printf("\n>> N = %d\n", N);

	while ((detValue == 0) || (isinf(detValue)))
	{
		if(detValue == 0)
			printf("\t\t\n\n *** Singular matrix, can't find its inverse ***\n");
		if (isinf(detValue))
			printf("\t\t\n\n *** Infinite determinant, can't find its inverse ***\n");
		printf("\t\t\n\n >>> ...New Values Will be Loading... <<<\n");

		fillinMatrix(Matrix);

		printf("\n\n=== NEW INPUT Matrix ===\n");
		//printMatrix(Matrix, N, N);

		detValue = detMatGauss(Matrix, N);
		printf("\n>> New Determinant = %.4f\n", detValue);
	}

	printf("\n\n\t\t *** NON Singular matrix, go to find its inverse ***\n");

	//fprintf(pf_text, "dim = %d\tdet = %.4f\t", N, detValue);
	//fprintf(pf_value, "%d\t%.4f\t", N, detValue);

	//Inizializzazione Host Matrix
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			h_matrix[i * N + j] = Matrix[i][j];
		}
	}
	/* Variabili condivise tra kernels*/
	cudaMalloc((void**)&d_matrix, size_m);
	cudaMalloc((void**)&d_dim, sizeof(int));
	cudaMalloc((void**)&d_detValue, sizeof(float));

	cudaMemcpy(d_matrix, h_matrix, size_m, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_detValue, &detValue, sizeof(float), cudaMemcpyHostToDevice);

	/* --- Inverse Matrix GPU Side using vesion using La Place ---
	
	cudaMalloc((void**)&d_inverseStore, size_m);

	//cuda event for benchmarking
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start of Cuda event
	cudaEventRecord(start);

	InverseCofactor_GPU << <grid, block >> > (d_matrix, d_dim, d_detValue, d_inverseStore);

	//CPU stall
	cudaDeviceSynchronize();

	//Stop of Cuda event
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	// Copy result back to host
	cudaMemcpy(h_inversemat, d_inverseStore, size_m, cudaMemcpyDeviceToHost);

	//Print of the results
	printf("\n\n\t=== Inverse Matrix using GPU ===\n"); fflush(stdout);

	elapsedI = elapsed / 1000;
	printf("\n >> The (Cofactor) execution time on GPU was %d s %.2f ms using cuda events\n", elapsedI, (elapsed - (elapsedI * 1000)));

	fprintf(pf_text, "GPU 1 = %.4f\t", elapsed);
	fprintf(pf_value, "%.4f\t", elapsed);


	printMatrix2(h_inversemat,N);
	}*/ //Uncomment this section to run Kernel that use LaPlace


	/* --- Inverse Matrix GPU Side using Gauss (Kernel base) --- */

	printf("\n\n\t=== GPU side Inverse Matrix using det Gauss - versione base ===\n"); fflush(stdout);
	cudaMalloc((void**)&d_inverseStore2, size_m);

	//cuda event for benchmarking
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	//Start of Cuda event
	
	/* Analysis 1: size of matrix N variable and fixed number of thread 
	int res = 27;
	if (N > res)
	{

		dim3 block2(res, res);
		int g = sqrt( (N * N) / (res * res) );
		if (g * g != ((N * N) / (res * res))) 
		{
			dim3 grid2(g+1, g+1);
			cudaEventRecord(start2);
			InverseCofactor_DetGauss_GPU_2 << <grid2, block2 >> > (d_matrix, d_dim, d_detValue, d_inverseStore2);
		}			
		else
		{
			dim3 grid2(g, g);
			cudaEventRecord(start2);
			InverseCofactor_DetGauss_GPU_2 << <grid2, block2 >> > (d_matrix, d_dim, d_detValue, d_inverseStore2);
		}					
	}
	else 
	{
		dim3 block2(N, N);
		dim3 grid2(1, 1);
		cudaEventRecord(start2);
		InverseCofactor_DetGauss_GPU_2 << <grid2, block2 >> > (d_matrix, d_dim, d_detValue, d_inverseStore2);
	}*/
	

	// Analysis 2: Matrix fixed size N and variable thread (change grid and block step by step)
	
	//Bidimensional configuration of grid and block

	//dim3 block2(block_n, block_n);
	//dim3 grid2(grid_n, grid_n);

	//int tot_thread = block_n * block_n * grid_n * grid_n;
	//fprintf(pf_text, "grid = %d\tblock = %d\t", grid_n, block_n);
	//fprintf(pf_value, "%d\t%d\t", grid_n, block_n);

	//fprintf(pf_text, "nThread = %d\t", tot_thread);
	//fprintf(pf_value, "%d\t", tot_thread);
	//InverseCofactor_DetGauss_GPU << <grid2, block2 >> > (d_matrix, d_dim, d_detValue, d_inverseStore2);

	//Unidimensional configuration of grid and block

	int tot_thread = block_n * grid_n;
	//fprintf(pf_text, "nThread = %d\t", tot_thread);
	//fprintf(pf_value, "%d\t", tot_thread);

	dim3 block2(block_n);
	dim3 grid2(grid_n);

	cudaEventRecord(start2);
	InverseCofactor_DetGauss_GPU_minor << <grid2, block2 >> > (d_matrix, d_dim, d_detValue, d_inverseStore2);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
			fprintf(stderr, "ERROR: %s: %s\n", "Kernel - InverseCofactor_DetGauss_GPU - Execution Failed!", cudaGetErrorString(cudaStatus));
	}
	else
	{
		//CPU stall
		cudaDeviceSynchronize();

		//Stop of Cuda event
		cudaEventRecord(stop2);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&elapsed2, start2, stop2);

		// Copy result back to host
		cudaMemcpy(h_inversemat2, d_inverseStore2, size_m, cudaMemcpyDeviceToHost);

		//Print of the results
		elapsedI2 = elapsed2 / 1000;
		printf("\n >> The (Cofactor) execution time on GPU with Gauss was %d s %.2f ms using cuda events\n", elapsedI2, (elapsed2 - (elapsedI2 * 1000)));

		//fprintf(pf_text, "GPUg = %.4f\n", elapsed2);
		//fprintf(pf_value, "%.4f\n", elapsed2);

		//printMatrix2(h_inversemat2, N);	
	}


	

	/* --- Inverse Matrix GPU Side using Gauss (Kernel optimized) --- 
	cudaMalloc((void**)&d_inverseStore3, size_m);

	//cuda event for benchmarking
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	//Start of Cuda event
	cudaEventRecord(start3);

	// test N fisso setting 
	dim3 block3(block_n, block_n);
	dim3 grid3(grid_n, grid_n);

	InverseCofactor_DetGauss_GPU_2 << <grid3, block3 >> > (d_matrix, d_dim, d_detValue, d_inverseStore3);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s: %s\n", "Kernel - InverseCofactor_DetGauss_GPU_2 - Execution Failed!", cudaGetErrorString(cudaStatus));
	}
	else
	{
		cudaDeviceSynchronize();	//CPU stall

		//Stop of Cuda event
		cudaEventRecord(stop3);
		cudaEventSynchronize(stop3);
		cudaEventElapsedTime(&elapsed3, start3, stop3);

		// Copy result back to host
		cudaMemcpy(h_inversemat3, d_inverseStore3, size_m, cudaMemcpyDeviceToHost);

		//Print of the results
		printf("\n\n\t=== GPU Inverse Matrix using Gauss opt version ===\n"); fflush(stdout);

		elapsedI3 = elapsed3 / 1000;
		printf("\n >> The (Cofactor) execution time on GPU 3 with Gauss was %d s %.2f ms using cuda events\n", elapsedI3, (elapsed3 - (elapsedI3 * 1000)));

		//fprintf(pf_text, "GPUg2 = %.4f\t", elapsed3);
		//fprintf(pf_value, "%.4f\t", elapsed3);

		printMatrix2(h_inversemat3, N);
	}*/


	/* --- Inverse Matrix CPU Side using La Place --- 

	printf("\n\n\t=== Inverse Matrix using Cofactor ===\n"); fflush(stdout);

	before = clock();
	InverseCofactor(Matrix, N, detValue, inverseMat);
	after = clock();

	difference = after - before;
	msec = difference * 1000 / CLOCKS_PER_SEC;
	s = msec / 1000;
	msecF = msec - (s * 1000);

	printf("\n >> The (Cofactor) execution time on CPU was %d s %.3f ms using c timer\n", s, msecF); fflush(stdout);
	//printMatrix(inverseMat, N, N);

	fprintf(pf_text, "CPU = %d\t", msec);
	fprintf(pf_value, "%d\t", msec);
	*/

	/* ---  Inverse Matrix CPU Side using Gauss --- */
	
	printf("\n\n\t=== Inverse Matrix using CPU Cofactor with Gauss ===\n"); fflush(stdout);

	before = clock();
	InverseCofactor2(Matrix, N, detValue, inverseMat);
	after = clock();

	difference = after - before;
	msec = difference * 1000 / CLOCKS_PER_SEC;
	s = msec / 1000;
	msecF = msec - (s * 1000);

	printf("\n >> The (Cofactor) execution time on CPU with Gauss was %d s %.3f ms using c timer\n", s, msecF); fflush(stdout);
	//printMatrix(inverseMat, N, N);

	//fprintf(pf_text, "CPUg = %d\n", msec);
	//fprintf(pf_value, "%d\n", msec);
	

//} //end test for (50 iteration)


/* --- Clean Memory --- */
	free(Matrix); free(inverseMat); cudaFree(d_matrix); cudaFree(d_dim); cudaFree(d_detValue);  cudaFree(d_inverseStore2); //cudaFree(d_inverseStore3);//cudaFree(d_inverseStore);
	//fclose(pf_text); fclose(pf_value); //Uncomment to use file generation in testing phase

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;

}	/* ------------------------------ END main() ------------------------------ */


/*
 * _Functions_
 */

 /********** COFACTOR METHOD **********/

void
InverseCofactor(float** Matrix, int dim, float detValue, float** inverseStore)
{
	int sign = 1;

	float** sub_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		sub_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}

	float** cof_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		cof_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}

	float** transposed_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		transposed_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}


	/* --- Fase 1 --- */
	for (size_t row = 0; row < dim; row++)
	{
		for (size_t col = 0; col < dim; col++)
		{
			for (size_t i = 0; i < dim - 1; i++)
			{
				for (size_t j = 0; j < dim - 1; j++)
				{
					size_t sub_row = (i < row ? i : i + 1);
					size_t sub_col = (j < col ? j : j + 1);
					sub_Mat[i][j] = Matrix[sub_row][sub_col];
				}
			}

			sign = ((row + col) % 2 == 0) ? 1 : -1;
			cof_Mat[row][col] = (sign) * (detMat(sub_Mat, dim - 1));
		}
	}

	/* --- Fase 2 ---*/
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			transposed_Mat[j][i] = cof_Mat[i][j];
		}
	}

	/* --- Fase 3 --- */
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			inverseStore[i][j] = transposed_Mat[i][j] / detValue;
		}
	}

	//Clean Up
	free(transposed_Mat); free(sub_Mat); free(cof_Mat);

	return;
}

void
InverseCofactor2(float** Matrix, int dim, float detValue, float** inverseStore)
{
	int sign = 1;

	float** sub_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		sub_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}

	float** cof_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		cof_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}

	float** transposed_Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		transposed_Mat[i] = (float*)malloc(sizeof(float) * dim);
	}


	/* --- Fase 1 --- */
	for (size_t row = 0; row < dim; row++)
	{
		for (size_t col = 0; col < dim; col++)
		{
			for (size_t i = 0; i < dim - 1; i++)
			{
				for (size_t j = 0; j < dim - 1; j++)
				{
					size_t sub_row = (i < row ? i : i + 1);
					size_t sub_col = (j < col ? j : j + 1);
					sub_Mat[i][j] = Matrix[sub_row][sub_col];
				}
			}

			sign = ((row + col) % 2 == 0) ? 1 : -1;
			cof_Mat[row][col] = (sign) * (detMatGauss(sub_Mat, dim - 1));
		}
	}

	/* --- Fase 2 ---*/
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			transposed_Mat[j][i] = cof_Mat[i][j];
		}
	}

	/* --- Fase 3 --- */
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			inverseStore[i][j] = transposed_Mat[i][j] / detValue;
		}
	}

	//Clean Up
	free(transposed_Mat); free(sub_Mat); free(cof_Mat);

	return;
}


/********** FUNCTIONS UTILITIES **********/

void
fillinMatrix(float** Matrix)
{
	int randomValues;
	float finalValue;

	printf("\t\t *** Loading of Values in Progress ***\n");

	srand((unsigned)time(NULL)); // -- RNG by current epoch -- 

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			randomValues = (rand() % 21) - 11; // Values takes in [-x,y]
			finalValue = (float)randomValues / 10; // Final values takes in [-1,1]

			Matrix[i][j] = finalValue;
		}
	}

	return;
}


void
printMatrix(float** Matrix, int n, int m)
{
	for (size_t i = 0; i < n; i++)
	{
		printf("\n");
		for (size_t j = 0; j < m; j++)
		{
			printf("%.4f  ", Matrix[i][j]);
		}
		printf("\n");
	}

	return;
}


void
printMatrix2(float* Matrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("\n");
		for (int j = 0; j < n; j++)
		{
			printf("%.4f ", Matrix[i * n + j]);
		}
		printf("\n");
	}
	return;
}

float
detMat(float** Matrix, int dim)
{
	float detValue = 0;

	float** sub_m = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		sub_m[i] = (float*)malloc(sizeof(float) * dim);
	}

	//Cardinalità uno
	if (dim == 1)
	{
		detValue = Matrix[0][0];
	}

	//Cardinalità due
	if (dim == 2)
	{
		detValue = Matrix[1][1] * Matrix[0][0] - Matrix[0][1] * Matrix[1][0];

	}
	else { //Cardinalità > 2

		for (size_t row = 0; row < dim; row++)
		{
			//Sottomatrice di ordine car-1
			for (size_t i = 0; i < dim - 1; i++)
			{
				for (size_t j = 0; j < dim - 1; j++)
				{
					size_t sub_row = (i < row ? i : i + 1);
					size_t sub_col = j + 1;
					sub_m[i][j] = Matrix[sub_row][sub_col];
				}
			}

			//Segno sottomatrice + per pari, - per dispari
			if (row % 2 == 0)
			{
				detValue += Matrix[row][0] * detMat(sub_m, dim - 1);

			}
			else {

				detValue -= Matrix[row][0] * detMat(sub_m, dim - 1);
			}
		}
	} // -- End external else --

	free(sub_m);

	return detValue;
}

float
detMatGauss(float** Matrix, int dim)
{

	/* --- Dichiarazione e/o inizializzazione Variabili --- */
  //Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValueerminante alla fine); swapCont: Contatore di Scambio delle righe
	float detValue = 1, Max, m;
	int swapCont = 0, index;

	/* --- Creazione Copia Matrice --- */
	float** Mat = (float**)malloc(sizeof(float) * dim * dim);
	for (size_t i = 0; i < dim; i++)
	{
		Mat[i] = (float*)malloc(sizeof(float) * dim);
	}
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			Mat[i][j] = Matrix[i][j];
		}
	}

	/*** FASE 1 ***/

/* --- Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione --- */

	for (size_t k = 0; k < dim - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
	{
		/* --- ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima --- */
		Max = fabs(Mat[k][k]);
		index = k;

		/* --- Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna --- */
		for (size_t i = k + 1; i < dim; i++)
		{
			if (fabs(Mat[i][k]) > Max)
			{
				Max = fabs(Mat[i][k]);
				index = i;
			}
		}

		/* --- Entra qui e controlla se Max si è aggiornato valutando index ed eventualmente effettuare il PIVOTING (PARZIALE) scambiando le righe tra loro --- */
		if (index != k)
		{
			swapCont++;

			//scambio di 2 vettori riga
			float* t = Mat[k];
			Mat[k] = Mat[index];
			Mat[index] = t;

			/*for(size_t j=k; j<n; j++)    //Scambio delle righe ma meno efficiente
			{
				aux=Mat[k][j];
				Mat[k][j]=Mat[index][j];
				Mat[index][j]=aux;
			}*/
		}

		/*** FASE 2 ***/

/* --- Applicazione del Metodo MEG per la Triangolarizzazione --- */
		for (size_t i = k + 1; i < dim; i++)
		{
			m = -Mat[i][k] / Mat[k][k];

			for (size_t j = k + 1; j < dim; j++)
			{
				Mat[i][j] = Mat[i][j] + (m * Mat[k][j]);
			}
		}
	}

	/*** FASE 3 ***/

//calcolo del determinante della matrice come prodotto degli elementi sulla diagonale
	for (size_t k = 0; k < dim; k++)
	{
		detValue *= Mat[k][k];
	}

	//per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segnio in base ed eventuali scambi)
	detValue *= ((swapCont & 1) ? -1.0 : 1.0);

	//printf("\nScambi Effettuati: %d", swapCont);

	return detValue;
}



/********** FUNCTIONS GPU SIDE (Kernel base - Versione det ricorsivo) **********/

__device__
void detMat_GPU(float* Matrix, int dim, float* det) {
	*det = 0;
	int size = (dim - 1) * (dim - 1) * sizeof(float);

	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	//Cardinalità uno
	if (dim == 1)
	{
		//printf("%d , %d : dim %d\n", row, col, dim);
		*det = Matrix[0];
	}

	//Cardinalità due
	if (dim == 2)
	{
		*det = Matrix[dim + 1] * Matrix[0] - Matrix[1] * Matrix[dim];
	}

	//Cardinalità tre
	if (dim == 3)
	{
		*det = (Matrix[0 * dim + 0] * Matrix[1 * dim + 1] * Matrix[2 * dim + 2] +
			Matrix[0 * dim + 1] * Matrix[1 * dim + 2] * Matrix[2 * dim + 0] +
			Matrix[0 * dim + 2] * Matrix[1 * dim + 0] * Matrix[2 * dim + 1])
			-
			(Matrix[0 * dim + 2] * Matrix[1 * dim + 1] * Matrix[2 * dim + 0] +
				Matrix[0 * dim + 0] * Matrix[1 * dim + 2] * Matrix[2 * dim + 1] +
				Matrix[0 * dim + 1] * Matrix[1 * dim + 0] * Matrix[2 * dim + 2]);
	}
	else { //Cardinalità > 3
		float* sub_m = new float[(dim - 1) * (dim - 1)];
		for (int row = 0; row < dim; row++)
		{
			//Sottomatrice di ordine car-1
			for (int i = 0; i < (dim - 1); i++)
			{
				for (int j = 0; j < (dim - 1); j++)
				{
					int sub_row = (i < row ? i : i + 1);
					int sub_col = j + 1;
					sub_m[i * (dim - 1) + j] = Matrix[sub_row * dim + sub_col];
				}
			}

			//Segno sottomatrice + per pari, - per dispari
			if (row % 2 == 0)
			{
				float det_subMat;
				detMat_GPU(sub_m, (dim - 1), &det_subMat);
				*det += Matrix[row * dim + 0] * det_subMat;

			}
			else {
				float det_subMat;
				detMat_GPU(sub_m, (dim - 1), &det_subMat);
				*det -= Matrix[row * dim + 0] * det_subMat;
			}

		}
		delete[] sub_m;
	} // -- End external else --

}

__global__
void InverseCofactor_GPU(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int sign = 1;

	float sub_Mat[(N - 1) * (N - 1)];

	__shared__ float cof_Mat[N * N];
	__shared__ float transposed_Mat[N * N];


	// Fase 1
	for (int i = 0; i < N - 1; i++)
	{
		for (int j = 0; j < N - 1; j++)
		{
			int sub_row = (i < row ? i : i + 1);
			int sub_col = (j < col ? j : j + 1);
			sub_Mat[i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
		}
	}

	sign = ((row + col) % 2 == 0) ? 1 : -1;
	float det_subMat;
	detMat_GPU(sub_Mat, *dim - 1, &det_subMat);

	cof_Mat[row * *dim + col] = (sign)*det_subMat;

	// Synchronize (ensure all the data is available)
	__syncthreads();

	// Fase 2
	transposed_Mat[col * *dim + row] = cof_Mat[row * *dim + col];

	// Fase 3
	inverseStore[row * *dim + col] = transposed_Mat[row * *dim + col] / *detValue;

	return;
}


/********** FUNCTIONS GPU SIDE (Kernel base - versione det Gauss) **********/

__global__
void InverseCofactor_DetGauss_GPU(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if((row < N) && (col < N)) //Overflow check
	{
		int sign = 1;

		float sub_Mat[(N - 1) * (N - 1)];
		float cof_Mat;
		float transposed_Mat;

		// Fase 1 : cofactor
		for (int i = 0; i < N - 1; i++)
		{
			for (int j = 0; j < N - 1; j++)
			{
				int sub_row = (i < row ? i : i + 1);
				int sub_col = (j < col ? j : j + 1);
				sub_Mat[i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
			}
		}

		sign = ((row + col) % 2 == 0) ? 1 : -1;

		float det_subMat = 1;

		//Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValue alla fine); swapCont: Contatore di Scambio delle righe
		float Max, m;
		int swapCont = 0, index;
		int dims = *dim - 1;


		// FASE 1 : determinant

		// Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione
		for (size_t k = 0; k < dims - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
		{
			// Ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima
			Max = fabs(sub_Mat[k * dims + k]);
			index = k;

			// Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna
			for (size_t i = k + 1; i < dims; i++)
			{
				if (fabs(sub_Mat[i * dims + k]) > Max)
				{
					Max = fabs(sub_Mat[i * dims + k]);
					index = i;
				}
			}

			// Effettua il PIVOTING (PARZIALE) scambiando le righe tra loro
			if (index != k)
			{
				swapCont++;

				float aux;
				for (size_t j = k; j < dims; j++)   
				{
					aux = sub_Mat[k * dims + j];
					sub_Mat[k * dims + j] = sub_Mat[index * dims + j];
					sub_Mat[index * dims + j] = aux;
				}
			}

			// FASE 2 : determinant
			// Applicazione del Metodo MEG per la Triangolarizzazione 

			for (size_t i = k + 1; i < dims; i++)
			{
				m = -sub_Mat[i * dims + k] / sub_Mat[k * dims + k];

				for (size_t j = k + 1; j < dims; j++)
				{
					sub_Mat[i * dims + j] = sub_Mat[i * dims + j] + (m * sub_Mat[k * dims + j]);
				}
			}
		}

		// FASE 3 : determinant
		// Calcolo del determinante della matrice come prodotto degli elementi sulla diagonale

		for (size_t k = 0; k < dims; k++)
		{
			det_subMat *= sub_Mat[k * dims + k];
		}

		// per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segno in base ed eventuali scambi)
		det_subMat *= ((swapCont & 1) ? -1.0 : 1.0);

		// fine calcolo determinante sottomatrice 

		cof_Mat = (sign)*det_subMat;

		// Fase 2/3 : cofactor
		inverseStore[col * *dim + row] = cof_Mat / *detValue;
	}
	return;
}

/********** FUNCTIONS GPU SIDE (Kernel ottimizzato - versione det Gauss) **********/

__global__
void InverseCofactor_DetGauss_GPU_2(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if ((row < N) && (col < N)) // overflow check
	{
		int sign = 1;

		float sub_Mat[(N - 1) * (N - 1)];
		float cof_Mat;
		float transposed_Mat;

		// Fase 1 : cofactor
		for (int i = 0; i < N - 1; i++)
		{
			for (int j = 0; j < N - 1; j++)
			{
				int sub_row = (i < row ? i : i + 1);
				int sub_col = (j < col ? j : j + 1);
				sub_Mat[i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
			}
		}

		sign = ((row + col) % 2 == 0) ? 1 : -1;

		float det_subMat = 1;

		//Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValueerminante alla fine); swapCont: Contatore di Scambio delle righe
		float Max, m;
		int swapCont = 0, index;
		int dims = *dim - 1;


		// FASE 1 : determinant

		// Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione 
		for (size_t k = 0; k < dims - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
		{
			if(sub_Mat[k * dims + k] == 0){ //condizione alternanza pivoting parziale (ottimizzazione : non ad ogni passo)
				// ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima
				Max = fabs(sub_Mat[k * dims + k]);
				index = k;

				// Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna
				for (size_t i = k + 1; i < dims; i++)
				{
					if (fabs(sub_Mat[i * dims + k]) > Max)
					{
						Max = fabs(sub_Mat[i * dims + k]);
						index = i;
					}
				}

				// Effettua il PIVOTING (PARZIALE) scambiando le righe tra loro
				if (index != k)
				{
					swapCont++;

					float aux;
					for (size_t j = k; j < dims; j++)
					{
						aux = sub_Mat[k * dims + j];
						sub_Mat[k * dims + j] = sub_Mat[index * dims + j];
						sub_Mat[index * dims + j] = aux;
					}
				}
			}

			// FASE 2 : determinant
			// Applicazione del Metodo MEG per la Triangolarizzazione

			for (size_t i = k + 1; i < dims; i++)
			{
				m = -sub_Mat[i * dims + k] / sub_Mat[k * dims + k];

				for (size_t j = k + 1; j < dims; j++)
				{
					sub_Mat[i * dims + j] = sub_Mat[i * dims + j] + (m * sub_Mat[k * dims + j]);
				}
			}
		}

		// FASE 3 : determinant
		//calcolo del determinante della matrice come prodotto degli elementi sulla diagonale
		for (size_t k = 0; k < dims; k++)
		{
			det_subMat *= sub_Mat[k * dims + k];
		}

		//per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segno in base ed eventuali scambi)
		det_subMat *= ((swapCont & 1) ? -1.0 : 1.0);

		// fine calcolo determinante sottomatrice

		cof_Mat = (sign)*det_subMat;

		// Fase 2/3 : cofactor 
		inverseStore[col * *dim + row] = cof_Mat / *detValue;
	}
	return;
}


/********** FUNCTIONS GPU SIDE (Kernel base - versione det Gauss USE unidimesional configuration of grid and block) **********/
__global__
void InverseCofactor_DetGauss_GPU_minor(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int id_thread = threadIdx.x + blockDim.x * blockIdx.x;;
	int row = id_thread/ *dim;
	int col = id_thread - (row* *dim);

	while ((row < N) && (col < N)) //overflow check
	{
		int sign = 1;

		float sub_Mat[(N - 1) * (N - 1)];
		float cof_Mat;
		float transposed_Mat;

		// Fase 1 : cofactor
		for (int i = 0; i < N - 1; i++)
		{
			for (int j = 0; j < N - 1; j++)
			{
				int sub_row = (i < row ? i : i + 1);
				int sub_col = (j < col ? j : j + 1);
				sub_Mat[i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
			}
		}

		sign = ((row + col) % 2 == 0) ? 1 : -1;

		float det_subMat = 1;

		//Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValueerminante alla fine); swapCont: Contatore di Scambio delle righe
		float Max, m;
		int swapCont = 0, index;
		int dims = *dim - 1;


		// FASE 1 : determinant

		// Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione
		for (size_t k = 0; k < dims - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
		{
			// ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima 
			Max = fabs(sub_Mat[k * dims + k]);
			index = k;

			// Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna 
			for (size_t i = k + 1; i < dims; i++)
			{
				if (fabs(sub_Mat[i * dims + k]) > Max)
				{
					Max = fabs(sub_Mat[i * dims + k]);
					index = i;
				}
			}

			// Effettua il PIVOTING (PARZIALE) scambiando le righe tra loro 
			if (index != k)
			{
				swapCont++;

				float aux;
				for (size_t j = k; j < dims; j++)
				{
					aux = sub_Mat[k * dims + j];
					sub_Mat[k * dims + j] = sub_Mat[index * dims + j];
					sub_Mat[index * dims + j] = aux;
				}
			}

			//FASE 2 : determinant
			//Applicazione del Metodo MEG per la Triangolarizzazione

			for (size_t i = k + 1; i < dims; i++)
			{
				m = -sub_Mat[i * dims + k] / sub_Mat[k * dims + k];

				for (size_t j = k + 1; j < dims; j++)
				{
					sub_Mat[i * dims + j] = sub_Mat[i * dims + j] + (m * sub_Mat[k * dims + j]);
				}
			}
		}

		// FASE 3 : determinant
		//calcolo del determinante della matrice come prodotto degli elementi sulla diagonale

		for (size_t k = 0; k < dims; k++)
		{
			det_subMat *= sub_Mat[k * dims + k];
		}

		//per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segno in base ed eventuali scambi)
		det_subMat *= ((swapCont & 1) ? -1.0 : 1.0);

		// fine calcolo determinante sottomatrice 

		cof_Mat = (sign)*det_subMat;

		// Fase 2/3 : cofactor
		inverseStore[col * *dim + row] = cof_Mat / *detValue;

		// calcolo nuovo elemento da analizzare (se ho meno thread rispetto elementi della matrice)
		id_thread = id_thread + blockDim.x * gridDim.x;
		row = id_thread / *dim;
		col = id_thread - (row * *dim);
	}
	return;
}

/********** FUNCTIONS GPU SIDE (Kernel ottimizzato - versione det Gauss USE unidimesional configuration of grid and block) **********/

__global__
void InverseCofactor_DetGauss_GPU_2_minor(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int id_thread = threadIdx.x + blockDim.x * blockIdx.x;;
	int row = id_thread / *dim;
	int col = id_thread - (row * *dim);

	while ((row < N) && (col < N)) //Overflow check
	{
		int sign = 1;

		float sub_Mat[(N - 1) * (N - 1)];
		float cof_Mat;
		float transposed_Mat;

		// Fase 1 : cofactor
		for (int i = 0; i < N - 1; i++)
		{
			for (int j = 0; j < N - 1; j++)
			{
				int sub_row = (i < row ? i : i + 1);
				int sub_col = (j < col ? j : j + 1);
				sub_Mat[i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
			}
		}

		sign = ((row + col) % 2 == 0) ? 1 : -1;

		float det_subMat = 1;

		//Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValueerminante alla fine); swapCont: Contatore di Scambio delle righe
		float Max, m;
		int swapCont = 0, index;
		int dims = *dim - 1;


		// FASE 1 : determinant
		// Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione

		for (size_t k = 0; k < dims - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
		{
			if (sub_Mat[k * dims + k] == 0) { //condizione alternanza pivoting parziale (ottimizzazione : non ad ogni passo)
				// ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima 
				Max = fabs(sub_Mat[k * dims + k]);
				index = k;

				// Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna 
				for (size_t i = k + 1; i < dims; i++)
				{
					if (fabs(sub_Mat[i * dims + k]) > Max)
					{
						Max = fabs(sub_Mat[i * dims + k]);
						index = i;
					}
				}

				// Effettua il PIVOTING (PARZIALE) scambiando le righe tra loro
				if (index != k)
				{
					swapCont++;

					float aux;
					for (size_t j = k; j < dims; j++)
					{
						aux = sub_Mat[k * dims + j];
						sub_Mat[k * dims + j] = sub_Mat[index * dims + j];
						sub_Mat[index * dims + j] = aux;
					}
				}
			}

			// FASE 2 : determinant
			//Applicazione del Metodo MEG per la Triangolarizzazione 

			for (size_t i = k + 1; i < dims; i++)
			{
				m = -sub_Mat[i * dims + k] / sub_Mat[k * dims + k];

				for (size_t j = k + 1; j < dims; j++)
				{
					sub_Mat[i * dims + j] = sub_Mat[i * dims + j] + (m * sub_Mat[k * dims + j]);
				}
			}
		}

		// FASE 3 : determinant
		//calcolo del determinante della matrice come prodotto degli elementi sulla diagonale

		for (size_t k = 0; k < dims; k++)
		{
			det_subMat *= sub_Mat[k * dims + k];
		}

		//per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segno in base ed eventuali scambi)
		det_subMat *= ((swapCont & 1) ? -1.0 : 1.0);

		//fine calcolo determinante sottomatrice 

		cof_Mat = (sign)*det_subMat;

		// Fase 2/3 : cofactor
		inverseStore[col * *dim + row] = cof_Mat / *detValue;

		// calcolo nuovo elemento da analizzare (se ho meno thread rispetto elementi della matrice)
		id_thread = id_thread + blockDim.x * gridDim.x;
		row = id_thread / *dim;
		col = id_thread - (row * *dim);
	}
	return;
}


/********** FUNCTIONS GPU SIDE (Kernel base - versione det Gauss) MINOR shared*********

__global__
void InverseCofactor_DetGauss_GPU_minorshared(float* Matrix, int* dim, float* detValue, float* inverseStore) {
	int id_thread = threadIdx.x + blockDim.x * blockIdx.x;;
	int row = id_thread / *dim;
	int col = id_thread - (row * *dim);

	while ((row < N) && (col < N)) //overflow check
	{
		int sign = 1;

		__shared__ float sub_Mat[cBlockdim][(N - 1) * (N - 1)];
		float cof_Mat;
		float transposed_Mat;

		//fase 1 : cofactor
		for (int i = 0; i < N - 1; i++)
		{
			for (int j = 0; j < N - 1; j++)
			{
				int sub_row = (i < row ? i : i + 1);
				int sub_col = (j < col ? j : j + 1);
				sub_Mat[threadIdx.x][i * (N - 1) + j] = Matrix[sub_row * *dim + sub_col];
			}
		}

		sign = ((row + col) % 2 == 0) ? 1 : -1;

		float det_subMat = 1;

		//Max: Contenitore valore Pivot massimo ricercato; m: Moltiplicatore di Gauss (Utile per il detValueerminante alla fine); swapCont: Contatore di Scambio delle righe
		float Max, m;
		int swapCont = 0, index;
		int dims = *dim - 1;

		//FASE 1 : determinant
		//Verifica delle CONDIZIONI DI APPLICABILITA' del Metodo di Eliminazione di Gauss, ed eventuale sua manipolazione

		for (size_t k = 0; k < dims - 1; k++)    //Ciclo che fa le (n-1) iterazioni necessarie
		{
			//ricerca elemento pivotale massimo: scambio la riga che lo contiene con la riga k-esima
			Max = fabs(sub_Mat[threadIdx.x][k * dims + k]);
			index = k;

			//Entra qui e controlla se Max è effettivamente un Max, altrimenti lo aggiorna
			for (size_t i = k + 1; i < dims; i++)
			{
				if (fabs(sub_Mat[threadIdx.x][i * dims + k]) > Max)
				{
					Max = fabs(sub_Mat[threadIdx.x][i * dims + k]);
					index = i;
				}
			}

			//Entra qui e controlla se Max si è aggiornato valutando index ed eventualmente effettuare il PIVOTING (PARZIALE) scambiando le righe tra loro
			if (index != k)
			{
				swapCont++;

				float aux;
				for (size_t j = k; j < dims; j++)
				{
					aux = sub_Mat[threadIdx.x][k * dims + j];
					sub_Mat[threadIdx.x][k * dims + j] = sub_Mat[threadIdx.x][index * dims + j];
					sub_Mat[threadIdx.x][index * dims + j] = aux;
				}
			}

			//FASE 2 : determinant 
			//Applicazione del Metodo MEG per la Triangolarizzazione

			for (size_t i = k + 1; i < dims; i++)
			{
				m = -sub_Mat[threadIdx.x][i * dims + k] / sub_Mat[threadIdx.x][k * dims + k];

				for (size_t j = k + 1; j < dims; j++)
				{
					sub_Mat[threadIdx.x][i * dims + j] = sub_Mat[threadIdx.x][i * dims + j] + (m * sub_Mat[threadIdx.x][k * dims + j]);
				}
			}
		}

		// FASE 3 : determinant
		//calcolo del determinante della matrice come prodotto degli elementi sulla diagonale

		for (size_t k = 0; k < dims; k++)
		{
			det_subMat *= sub_Mat[threadIdx.x][k * dims + k];
		}

		//per ogni scambio di righe effettuato moltiplico il determinante per -1 (Gestione segno in base ed eventuali scambi)
		det_subMat *= ((swapCont & 1) ? -1.0 : 1.0);

		// fine calcolo determinante sottomatrice 

		cof_Mat = (sign)*det_subMat;

		// Fase 2/3 : cofactor
		inverseStore[col * *dim + row] = cof_Mat / *detValue;

		// calcolo nuovo elemento da analizzare (se ho meno thread rispetto elementi della matrice)
		id_thread = id_thread + blockDim.x * gridDim.x;
		row = id_thread / *dim;
		col = id_thread - (row * *dim);
	}
	return;
}

*/