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
	int pad_width = pool_width/2;
	int pad_height = pool_height/2;
	int block_x_index = blockDim.x*blockIdx.x;
	int block_y_index = blockDim.y*blockIdx.y;

	int global_offset = blockIdx.z*image_width*image_height;
	// int global_offset = 0;
	int loc_x_index = threadIdx.x;
	int loc_y_index = threadIdx.y;

	int global_x_index = block_x_index + loc_x_index;
	int global_y_index = block_y_index + loc_y_index;

	for(int i = threadIdx.x - pad_height; i <= blockDim.x + pad_height; i = i + blockDim.x){
		for(int j = threadIdx.y - pad_width; j <= blockDim.y + pad_width; j = j + blockDim.y){
			int shared_mem_index = (i+pad_height)*(blockDim.y+pad_width) + (j+pad_width);
			global_x_index = block_x_index + i;
			global_y_index = block_y_index + j;
			shared_pointer[shared_mem_index] = 0;
			if(global_x_index < 0 || global_x_index >= image_height || global_y_index < 0 || global_y_index >= image_width){
				shared_pointer[shared_mem_index] = 0;
			}else{
				shared_pointer[shared_mem_index] = global_pointer[global_offset+ (global_x_index*WIDTH) + global_y_index];
			}
		}
	}
	__syncthreads();
	
	double max_value = shared_pointer[(threadIdx.x)*(blockDim.y + pad_width) + threadIdx.y];
	for(int i = 0; i < pool_height; i++){
		for(int j = 0; j < pool_width; j++){
			int loc_index = (i+threadIdx.x)*(blockDim.y + pad_width) + (j+threadIdx.y);
				if(shared_pointer[loc_index] > max_value){
					max_value = shared_pointer[loc_index];
			}
		}
	}
	output_pointer[0] = 122.0;
	// output_pointer[global_offset+global_x_index*blockDim.y + global_y_index] = max_value;

}
int get_shared_memory_size(int pooling_height, int pooling_width){
	int total_height = TILE_HEIGHT + pooling_height + 1;
	int total_width = TILE_WIDTH + pooling_width + 1;
	return total_width*total_height;
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
  	max_pool_kernel<<<image_grid_vector, image_block_vector, shared_memory_size>>>(CHANNELS, HEIGHT, WIDTH, pooling_height, pooling_width, gpu_image_pointer, gpu_output_pointer);
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



