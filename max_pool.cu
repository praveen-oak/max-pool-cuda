#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <cudnn.h>


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

//order of coordinates is X(height), Y(width), channel
#define Y_STRIDE_SIZE 1 //number of channels
#define X_STRIDE_SIZE 1024	//max_width*channels
#define CHANNEL_STRIDE_SIZE 1024*1024
#define DIV_RUP(x, y)	(((x)+(y)-1)/(y))

__global__ void max_pool_kernel(int channels, int image_height, int image_width, int pool_height, int pool_width,
		double *global_pointer, double *output_pointer)
{
	extern __shared__ double shared_pointer[];
	output_pointer[0] = 10000.0;
	// int pad_width = pool_width/2;
	// int pad_height = pool_height/2;
	// int block_x_index = blockDim.x*blockIdx.x;
	// int block_y_index = blockDim.y*blockIdx.y;

	// int loc_x_index = threadIdx.x;
	// int loc_y_index = threadIdx.y;

	// int global_x_index = block_x_index + loc_x_index;
	// int global_y_index = block_y_index + loc_y_index;


	// int loc_start_index = loc_y_index + loc_x_index*(blockDim.y);
	// int stride = (blockDim.x)*local_x_stride;
	// shared_pointer[loc_s]

	
	// output_pointer[0] = shared_pointer[1];
	// loc_

	// for(int k = 0; k < channels ; k++){
	// 	for(int i = thread_x_start_co_ordinate;i < block_x_end_co_ordinate;i = i + blockDim.x){
	// 		for(int j = thread_y_start_co_ordinate;j < block_y_end_co_ordinate;j = j + blockDim.y){
	// 			int loc_index = i*local_x_stride + j*local_y_stride + k*local_channel_stride;
	// 			if(loc_index > 3072){
	// 				continue;
	// 			}else{
	// 				int global_x_index = i+block_x_co_ordinate-pad_height/2;
	// 				int global_y_index = j+block_y_co_ordinate-pad_width/2;
	// 				if(global_x_index < 0 || global_x_index > image_height || global_y_index < 0 || global_y_index > image_width){
	// 					shared_pointer[loc_index] = 0;
	// 				}else{
	// 					int global_index = global_x_index*X_STRIDE_SIZE + global_y_index*Y_STRIDE_SIZE + k*CHANNEL_STRIDE_SIZE;
	// 					shared_pointer[loc_index] = global_pointer[global_index];	
	// 				}
	// 			}
				
	// 		}
	// 	}
	// }

	// __syncthreads();
	
	// output_pointer[0] = 1000.0;
	// for(int k = 0; k < channels; k++){
	// 	int max_value = 0;
	// 	int value;
	// 	for(int i = 0; i < pool_height; i++){
	// 		for(int j = 0; i < pool_width; j++){
	// 			value = shared_pointer[k*local_channel_stride + i*local_x_stride + j*local_y_stride];
	// 			if(value > max_value){
	// 				max_value = value;	
	// 			}
	// 		}
	// 	}
	// 	output_pointer[k*CHANNEL_STRIDE_SIZE + threadIdx.x*X_STRIDE_SIZE + threadIdx.y*Y_STRIDE_SIZE] = max_value;
	// }
	// __syncthreads();

}
int get_shared_memory_size(int pooling_height, int pooling_width){
	int total_height = TILE_HEIGHT;
	int total_width = TILE_WIDTH;
	return CHANNELS*total_width*total_height;
}

///////////////////////////////////////////////////////////////////////////////
// Create Image in CPU memory
////////////////////////////////////////////////////////////////////////////////
void fill_image(int channels, int height, int width, double *image_pointer)
{
  int image_memory_size = channels*height*width*sizeof(double);
  memset(image_pointer, 0, image_memory_size);
  for(int k = 0; k < channels; k++){
  	for(int i = 0; i < height; i++){
  		for(int j = 0; j < width; j++){
  			int index = i*X_STRIDE_SIZE + j*Y_STRIDE_SIZE + k*CHANNEL_STRIDE_SIZE;
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
  				int index = i*X_STRIDE_SIZE + j*Y_STRIDE_SIZE + k*CHANNEL_STRIDE_SIZE;
  				sum = sum + image_pointer[index];
  				// printf("i = %d, j = %d, k = %d Value of data at index %d is %lf\n",i, j, k, index, image_pointer[index]);
  			}
  		}
  	}
  	printf("Check sum value is %lf \n",sum);
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
  				int index = i*X_STRIDE_SIZE + j*Y_STRIDE_SIZE + k*CHANNEL_STRIDE_SIZE;
  				sum = sum + output_pointer[index];
  			}
  		}
  	}
  	printf("The checksum after the max_pool is %lf \n",sum);
}
__global__ void test_gpu_copy(double *image_pointer, double *image_output_pointer){
	int x_coordinate = blockDim.x*blockIdx.x + threadIdx.x;
	int y_coordinate = blockDim.y*blockIdx.y + threadIdx.y;

	for(int k = 0;k < 3;k++){
		int index = x_coordinate*X_STRIDE_SIZE + y_coordinate*Y_STRIDE_SIZE + k*CHANNEL_STRIDE_SIZE;
		image_output_pointer[index] = image_pointer[index];
	}

}

int main(int ac, char *av[]){
	int image_size = CHANNELS*HEIGHT*WIDTH*sizeof(double);
	int pooling_height = 21;
	int pooling_width = 21;
	double *gpu_image_pointer, *gpu_output_pointer;
	double *image_pointer, *output_pointer;

	image_pointer = (double *) malloc(image_size);
	output_pointer = (double *) malloc(image_size);
	output_pointer[0] = 500.0;
  	fill_image(CHANNELS, HEIGHT, WIDTH, image_pointer);
  	validate_image_data(CHANNELS, HEIGHT, WIDTH, image_pointer);

  	CUDA_CALL(cudaMalloc(&gpu_image_pointer, image_size));
  	CUDA_CALL(cudaMalloc(&gpu_output_pointer, image_size));
  	CUDA_CALL(cudaMemcpy(gpu_image_pointer, image_pointer, image_size, cudaMemcpyHostToDevice));
  	cudaDeviceSynchronize();

  	dim3 image_block_vector(TILE_WIDTH, TILE_HEIGHT);
  	dim3 image_grid_vector(DIV_RUP(WIDTH, TILE_WIDTH), DIV_RUP(HEIGHT, TILE_HEIGHT), 3);

  	int shared_memory_size = get_shared_memory_size(pooling_height, pooling_width);
  	shared_memory_size = shared_memory_size*sizeof(double);
  	printf(" Shared memory size = %d\n", shared_memory_size);
  	max_pool_kernel<<<image_grid_vector, image_block_vector, 1>>>(CHANNELS, HEIGHT, WIDTH, pooling_height, pooling_width, gpu_image_pointer, gpu_output_pointer);
    cudaDeviceSynchronize();
    cudaMemcpy(output_pointer, gpu_output_pointer, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  	printf("After gpu code\n");
  	validate_image_data(CHANNELS, HEIGHT, WIDTH, output_pointer);


    
   //  print_max_pool_checksum(CHANNELS, HEIGHT, WIDTH, output_pointer);
  	// CUDA_CALL(cudaFree(gpu_image_pointer));
  	// CUDA_CALL(cudaFree(gpu_output_pointer));
  	// free(image_pointer);
  	// free(output_pointer);







  		// test_gpu_copy<<<image_grid_vector, image_block_vector>>>(gpu_image_pointer, gpu_output_pointer);
  	// cudaDeviceSynchronize();

  	// memset(output_pointer, 0, image_size);
  	// cudaMemcpy(output_pointer, gpu_output_pointer, image_size, cudaMemcpyDeviceToHost);
  	// cudaDeviceSynchronize();


}



