#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <cudnn.h>
#include <string.h>
#include <stdlib.h>



#define CUDA_CALL(x) do {						\
  cudaError_t ____rc = (x);					\
  assert(____rc == cudaSuccess);					\
} while (0)

/* Image channels, height, width. */
#define CHANNELS	  3
#define HEIGHT	  1024
#define WIDTH	  1024

/* Tile size. */
#define TILE_WIDTH		32
#define TILE_HEIGHT		32

#define POOL_WIDTH 21
#define POOL_HEIGHT 21

#define REPS		3
#define DIV_RUP(x, y)	(((x)+(y)-1)/(y))


__global__ void max_pool_kernel(double *global_pointer, double *output_pointer){

	int pad_height = POOL_HEIGHT/2;
	int pad_width = POOL_WIDTH/2;

	int shared_pointer_height = pad_height*2 + blockDim.y;
	int shared_pointer_width = pad_width*2 + blockDim.x;
	__shared__ double shared_pointer[3000];
	int channel_offset = blockIdx.z * HEIGHT * WIDTH;
	
	int global_x_offset = blockDim.x * blockIdx.x;
	int global_y_offset = blockDim.y * blockIdx.y;

	int global_x_val = 0;
	int global_y_val = 0;
	int shared_index = 0;
	int global_index = 0;

	int i = threadIdx.y;	//index of y
	int j = threadIdx.x;	//index of x

	while(i < shared_pointer_height){
		global_y_val = global_y_offset + i - pad_height;
		j = threadIdx.x;
		while(j < shared_pointer_width){
			shared_index = i*shared_pointer_width + j;
			global_x_val = global_x_offset + j - pad_width;
			double global_value = 0.0;
			if(global_y_val < 0 || global_y_val >= HEIGHT || global_x_val < 0 || global_x_val >= WIDTH){
				shared_pointer[shared_index] = 0.0;
			}else{
				global_index = channel_offset + global_y_val*WIDTH + global_x_val;
				global_value = global_pointer[global_index];
				shared_pointer[shared_index] = global_value;
				
			}
			j = j + blockDim.x;
		}
		i = i + blockDim.y;
	}

	 __syncthreads();

	channel_offset = blockIdx.z*HEIGHT*WIDTH;
	double max_val = 0.0;
	// int shared_memory_height = blockDim.y + (POOL_HEIGHT/2) * 2;
	int shared_memory_width = blockDim.x + (POOL_WIDTH/2) * 2;

	i = 0;
	j = 0;

	while(i < POOL_HEIGHT){
		j = 0;
		while(j < POOL_WIDTH){
			int x_offset = threadIdx.x + j;
			int y_offset = threadIdx.y + i;
			int shared_index = y_offset*shared_memory_width + x_offset;
			double temp_max = shared_pointer[shared_index];
			if(temp_max > max_val){
				max_val = temp_max;
			}
			j = j + 1;
		}
		i = i + 1;

	}
	int global_x_index = blockDim.x*blockIdx.x + threadIdx.x;
	int global_y_index = blockDim.y*blockIdx.y + threadIdx.y;

	global_index = channel_offset + global_y_index*WIDTH + global_x_index;
	output_pointer[global_index] = max_val;
  	
}

void fill_image(int channels, int height, int width, double *image_pointer)
{
  int image_memory_size = channels*height*width*sizeof(double);
  memset(image_pointer, 0, image_memory_size);
  for(int k = 0; k < channels; k++){
  	for(int i = 0; i < height; i++){
  		for(int j = 0; j < width; j++){
  			int index = i*WIDTH + j + k*WIDTH*HEIGHT;
  			image_pointer[index] = k*(i+j);
  		}
  	}
  }
}

void validate_image_data(int channels, int height, int width, double *image_pointer){
	double sum = 0.0;

	for(int k = 0; k < channels; k++)
		for(int i = 0; i < height; i++){
  			for(int j = 0; j < width; j++){
  			{
  				int index = i*WIDTH + j + k*WIDTH*HEIGHT;
  				sum = sum + image_pointer[index];
  			}
  		}
  	}
  	printf("Check sum of image data is is %lf \n",sum);
  	if(sum == 3218079744.0){
  		printf("Check sum of image validated \n");
  	}
  	else{
  		printf("Check sum is wrong.\n",sum);
  		printf("Exiting program \n");
  		exit(0);
  	}
}

void print_max_pool_checksum(int channels, int height, int width, double *output_pointer){
	double sum = 0.0;
	for(int k = 0; k < channels; k++)
		for(int i = 0; i < height; i++){
  			for(int j = 0; j < width; j++){
  			{
  				int index = i*WIDTH + j + k*WIDTH*HEIGHT;
  				sum = sum + output_pointer[index];
  			}
  		}
  	}
  	printf("The checksum of data after maxpool is %lf \n",sum);
}
void check_on_cpu(double *image_pointer, double *output_pointer){
	int pooling_height = POOL_HEIGHT;
	int pooling_width = POOL_WIDTH;

  	int index = 0;
  	int pad_height = pooling_height/2;
  	int pad_width = pooling_width/2;
  	int output_index = 0;
  	for(int c = 0; c < CHANNELS; c++){
  		int offset = c*HEIGHT*WIDTH;
  		for(int i = 0; i< HEIGHT; i++){
	  		for(int j = 0; j< WIDTH; j++){
	  			int start_i = i;
	  			int start_j = j;

	  			int max_val = 0;
	  			for(int k = start_i - pad_height; k <= start_i+pad_height; k++){
	  				for(int l = start_j - pad_width; l <= start_j+pad_width; l++){
	  					if(k >= 0 && k < HEIGHT && l >= 0 && l < WIDTH){
	  						index = offset + k*WIDTH + l;
	  						if(image_pointer[index] > max_val){
	  							max_val = image_pointer[index];
	  						}
	  					}
	  				}
	  			}
	  			output_index = offset + i*WIDTH + j;
	  			output_pointer[output_index] = max_val;
	  		}
	  	}
  	}
}

static double TimeSpecToSeconds(struct timespec* ts){
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}


#define checkCUDNN(expression)                    \
{                                                  \
	cudnnStatus_t status = (expression);           			  \
	if (status != CUDNN_STATUS_SUCCESS) {        			    \
		printf("cuDNN error on line %d: %s\n" , __LINE__ ,	\
				cudnnGetErrorString(status));  						\
		exit(EXIT_FAILURE);                           \
	}                                               \
}																									\


struct CuDNN_Setup { 
  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  size_t workspace_bytes;
  void* d_workspace;
  double alpha, beta;
};


void prepareCuDNN(CuDNN_Setup *pS, double *image_pointer, double *output_pointer) {
	/* Create context object*/
	checkCUDNN(cudnnCreate(&pS->cudnn));

	/* Describe input tensor*/
	checkCUDNN(cudnnCreateTensorDescriptor(&pS->input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(pS->input_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_DOUBLE,
				/*batch_size=*/1,
				/*channels=*/CHANNELS,
				/*image_height=*/HEIGHT,
				/*image_width=*/WIDTH));

	/* Describe output tensor*/
	checkCUDNN(cudnnCreateTensorDescriptor(&pS->output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(pS->output_descriptor,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*dataType=*/CUDNN_DATA_DOUBLE,
				/*batch_size=*/1,
				/*channels=*/CHANNELS,
				/*image_height=*/HEIGHT,
				/*image_width=*/WIDTH));


	/* Describe max pooling*/
	checkCUDNN(cudnnCreatePoolingDescriptor(&pS->pooling_descriptor));
	checkCUDNN(cudnnSetPooling2dDescriptor(
    /*cudnnPoolingDescriptor_t*/    				pS->pooling_descriptor,
    /*cudnnPoolingMode_t*/          				CUDNN_POOLING_MAX,
    /*cudnnNanPropagation_t*/      					CUDNN_NOT_PROPAGATE_NAN,
    /*height of pooling*/	                        POOL_HEIGHT,
    /*width of pooling */	                        POOL_WIDTH,
    /*vertical padding*/							POOL_HEIGHT/2,
    /*horizontal padding*/							POOL_WIDTH/2,
    /*vertical stride*/								1,
    /*horizontal stride*/							1));

	pS->alpha=1.0;
	pS->beta=0.0;
}


void CleanupCuDNN(CuDNN_Setup *pS) {


	/* Free workspace */
	cudaFree(pS->d_workspace);

	/* Destroy descriptors */
	cudnnDestroyTensorDescriptor(pS->input_descriptor);
	cudnnDestroyTensorDescriptor(pS->output_descriptor);
	cudnnDestroyPoolingDescriptor(pS->pooling_descriptor);
	cudnnDestroy(pS->cudnn);
}

void do_max_pooling(CuDNN_Setup *pS, double *image_pointer, double *output_pointer) {
	checkCUDNN(cudnnPoolingForward(
    /*cudnnHandle_t*/                    pS->cudnn,
    /*const cudnnPoolingDescriptor_t*/   pS->pooling_descriptor,
    /*const void*/                      &pS->alpha,
    /*const cudnnTensorDescriptor_t*/    pS->input_descriptor,
    /*const void*/                       image_pointer,
    /*const void*/                      &pS->alpha,
    /*const cudnnTensorDescriptor_t*/    pS->output_descriptor,
    /*void*/                             output_pointer));
}

void do_cu_dnn_maxpooling(){

	printf("Running the max pooling algorithm using CuDNN library \n\n");

	struct timespec start;
  	struct timespec end;
  	double copytime;
	int image_size = CHANNELS*HEIGHT*WIDTH*sizeof(double);
	double *gpu_image_pointer, *gpu_output_pointer;
	double *image_pointer, *output_pointer, *cpu_output_pointer;

	image_pointer = (double *) malloc(image_size);
	output_pointer = (double *) malloc(image_size);
	cpu_output_pointer = (double *) malloc(image_size);
	memset(output_pointer, 0, image_size);
	memset(cpu_output_pointer, 0, image_size);

  	fill_image(CHANNELS, HEIGHT, WIDTH, image_pointer);
  	validate_image_data(CHANNELS, HEIGHT, WIDTH, image_pointer);

  	CUDA_CALL(cudaMalloc(&gpu_image_pointer, image_size));
  	CUDA_CALL(cudaMalloc(&gpu_output_pointer, image_size));

  	CuDNN_Setup cuDNN_setup;
  	prepareCuDNN(&cuDNN_setup, gpu_image_pointer, gpu_output_pointer);


    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    CUDA_CALL(cudaMemcpy(gpu_image_pointer, image_pointer, image_size, cudaMemcpyHostToDevice));  
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy host->dev = %lf sec\n", copytime);

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    do_max_pooling(&cuDNN_setup, gpu_image_pointer, gpu_output_pointer);
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Time cudnn =  %lf sec\n", copytime);

    if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    CUDA_CALL(cudaMemcpy(output_pointer, gpu_output_pointer, image_size, cudaMemcpyDeviceToHost));                          // copy result back
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    printf("Copy dev->host kernel %lf sec\n", copytime);

    print_max_pool_checksum(CHANNELS, HEIGHT, WIDTH, output_pointer);


	CleanupCuDNN(&cuDNN_setup);
	CUDA_CALL(cudaFree(gpu_image_pointer));
	CUDA_CALL(cudaFree(gpu_output_pointer));
	free(image_pointer);
	free(output_pointer);
	printf("\n\n");

}

void do_cuda_max_pooling(){

	printf("Running the max pooling algorithm using shared memory \n\n");
	struct timespec start;
  	struct timespec end;
  	double copytime;
	int image_size = CHANNELS*HEIGHT*WIDTH*sizeof(double);
	double *gpu_image_pointer, *gpu_output_pointer;
	double *image_pointer, *output_pointer, *cpu_output_pointer;

	image_pointer = (double *) malloc(image_size);
	output_pointer = (double *) malloc(image_size);
	cpu_output_pointer = (double *) malloc(image_size);
	memset(output_pointer, 0, image_size);
	memset(cpu_output_pointer, 0, image_size);
  	fill_image(CHANNELS, HEIGHT, WIDTH, image_pointer);
  	validate_image_data(CHANNELS, HEIGHT, WIDTH, image_pointer);

  	CUDA_CALL(cudaMalloc(&gpu_image_pointer, image_size));
  	CUDA_CALL(cudaMalloc(&gpu_output_pointer, image_size));
  	if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
  	CUDA_CALL(cudaMemcpy(gpu_image_pointer, image_pointer, image_size, cudaMemcpyHostToDevice));
  	cudaDeviceSynchronize();
  	if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
  	copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
  	printf("Copy host->dev = %lf sec \n",copytime);

  	dim3 image_block_vector(TILE_WIDTH, TILE_HEIGHT);
  	dim3 image_grid_vector(DIV_RUP(WIDTH, TILE_WIDTH), DIV_RUP(HEIGHT, TILE_HEIGHT), CHANNELS);

  	if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
  	max_pool_kernel<<<image_grid_vector, image_block_vector>>>(gpu_image_pointer, gpu_output_pointer);
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
  	printf("Time cuda code = %lf sec \n",copytime);


  	if(clock_gettime(CLOCK_MONOTONIC, &start)){ printf("CLOCK ERROR"); }
    cudaMemcpy(output_pointer, gpu_output_pointer, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if(clock_gettime(CLOCK_MONOTONIC, &end)) { printf("CLOCK ERROR"); }
    copytime = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
  	printf("Copy dev->host = %lf sec \n",copytime);

    print_max_pool_checksum(CHANNELS, HEIGHT, WIDTH, output_pointer);


    CUDA_CALL(cudaFree(gpu_image_pointer));
    CUDA_CALL(cudaFree(gpu_output_pointer));
   	free(image_pointer);
   	free(output_pointer);
   	free(cpu_output_pointer);

   	printf("\n\n");

}
int main(int ac, char *av[]){
	do_cuda_max_pooling();
  	do_cu_dnn_maxpooling();
}



