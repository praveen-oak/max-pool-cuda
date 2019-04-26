#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <cudnn.h>


#define CUDA_CALL(x) do {						\
  cudaError_t ____rc = (x);					\
  assert(____rc == cudaSuccess);					\
} while (0)

/* Image channels, height, width. */
#define CHANNELS	  1
#define HEIGHT	  3
#define WIDTH	  3

/* Tile size. */
#define TILE_WIDTH		1
#define TILE_HEIGHT		1

#define POOL_WIDTH 3
#define POOL_HEIGHT 3

#define DIV_RUP(x, y)	(((x)+(y)-1)/(y))

__global__ void max_pool_kernel(int channels, int image_height, int image_width, int pool_height, int pool_width,
		double *global_pointer, double *output_pointer)
{
	__shared__ double shared_pointer[2704];
	int pad_width = pool_width/2;
	int pad_height = pool_height/2;
	int block_x_index = blockDim.x*blockIdx.x;
	int block_y_index = blockDim.y*blockIdx.y;

	int global_offset = blockIdx.z*image_width*image_height;
	int global_x_index;
	int global_y_index;

	int i = -1*pad_height;
	int j = -1*pad_width;
	int shared_mem_index = 0;
	// while(i < blockDim.y + 2*pad_height){
	// 	while(j < blockDim.x + 2*pad_width){
	// 		shared_mem_index = (i+pad_height)*(2*pad_width+blockDim.x) + (j+pad_width)
	// 		global_x_index = block_x_index + j;
	// 		global_y_index = block_y_index + i;
	// 		if(global_x_index < 0 || global_x_index >= image_width || global_y_index < 0 || global_y_index >= image_height){
	// 			shared_pointer[shared_mem_index] = 0;
	// 		}else{
	// 			shared_pointer[shared_mem_index] = global_pointer[global_offset+ (global_y_index*WIDTH) + global_x_index];
	// 		}
	// 	}
	// }

	for(int i = threadIdx.y; i < blockDim.y + 2*pad_height; i = i + blockDim.y){
		for(int j = threadIdx.x; j < blockDim.x + 2*pad_width; j = j + blockDim.x){
			int shared_mem_index = i*(blockDim.x+ 2*pad_width) + j;
			global_y_index = block_y_index - pad_height;
			global_x_index = block_x_index - pad_width;
			if(global_x_index < 0 || global_x_index >= image_width || global_y_index < 0 || global_y_index >= image_height){
				shared_pointer[shared_mem_index] = 0;
			}else{
				shared_pointer[shared_mem_index] = global_pointer[global_offset+ (global_y_index*WIDTH) + global_x_index];
			}
		}
	}
	__syncthreads();
	double max_value = 0.0;
	for(int i = 0; i < pool_height; i++){
		for(int j = 0; j < pool_width; j++){
			int loc_index = (i+threadIdx.y)*(blockDim.x + 2*pad_width) + (j+threadIdx.x);
			if(shared_pointer[loc_index] > max_value){
				max_value = shared_pointer[loc_index];
			}
		}
	}
	global_y_index = block_y_index + threadIdx.y;
	global_x_index = block_x_index + threadIdx.x;
	output_pointer[global_offset+ (global_y_index*WIDTH) + global_x_index] = max_value;
}
int get_shared_memory_size(int pooling_height, int pooling_width){
	int total_height = TILE_HEIGHT + pooling_height/2 * 2;
	int total_width = TILE_WIDTH + pooling_width/2 * 2;
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
  			int index = i*WIDTH + j + k*WIDTH*HEIGHT;
  			image_pointer[index] = (i+j);
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
  	printf("Check sum value is %lf \n",sum);
  	if(sum == 3218079744.0){
  		printf("Check sum of image validated \n");
  	}
  	else{
  		printf("Check sum is wrong.\n",sum);
  		printf("Exiting program \n");
  		// exit(0);
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
  	printf("The checksum after the max_pool is %lf \n",sum);
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

void print_image(double *image_pointer){
	for(int c = 0; c < CHANNELS; c++){
  		int offset = c*HEIGHT*WIDTH;
  		for(int i = 0; i< HEIGHT; i++){
  			for(int j = 0; j< WIDTH; j++){
  				int index = offset + i*WIDTH + j;
  				int cpu_value = image_pointer[index];
  				printf(" %d ",cpu_value);
  			}
  			printf("\n");
  		}
  		printf("\n\n");
  	}
}

int main(int ac, char *av[]){
	int image_size = CHANNELS*HEIGHT*WIDTH*sizeof(double);
	int pooling_height = POOL_HEIGHT;
	int pooling_width = POOL_WIDTH;
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
  	CUDA_CALL(cudaMemcpy(gpu_image_pointer, image_pointer, image_size, cudaMemcpyHostToDevice));
  	cudaDeviceSynchronize();

  	dim3 image_block_vector(TILE_WIDTH, TILE_HEIGHT);
  	dim3 image_grid_vector(DIV_RUP(WIDTH, TILE_WIDTH), DIV_RUP(HEIGHT, TILE_HEIGHT), CHANNELS);

  	int shared_memory_size = get_shared_memory_size(pooling_height, pooling_width);
  	shared_memory_size = shared_memory_size*sizeof(double);
  	printf(" Shared memory size = %d\n", shared_memory_size);
  	max_pool_kernel<<<image_grid_vector, image_block_vector>>>(CHANNELS, HEIGHT, WIDTH, pooling_height, pooling_width, gpu_image_pointer, gpu_output_pointer);
    cudaDeviceSynchronize();
    cudaMemcpy(output_pointer, gpu_output_pointer, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    print_max_pool_checksum(CHANNELS, HEIGHT, WIDTH, output_pointer);
    check_on_cpu(image_pointer, cpu_output_pointer);
    print_max_pool_checksum(CHANNELS, HEIGHT, WIDTH, cpu_output_pointer);

  	
}



