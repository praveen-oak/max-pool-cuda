#include <time.h>
#include <assert.h>
#include <stdio.h>
// #include <cudnn.h>
#include <string.h>
#include <stdlib.h>


// #define CUDA_CALL(x) do {						\
//   cudaError_t ____rc = (x);					\
//   assert(____rc == cudaSuccess);					\
// } while (0)

/* Image channels, height, width. */
#define CHANNELS	  1
#define HEIGHT	  9
#define WIDTH	  9

/* Tile size. */
#define TILE_WIDTH		3
#define TILE_HEIGHT		3

#define POOL_WIDTH 3
#define POOL_HEIGHT 3

#define DIV_RUP(x, y)	(((x)+(y)-1)/(y))


struct BlockDim{
	int x;
	int y;
	int z;
};

struct BlockIdx{
	int x;
	int y;
	int z;
};

struct ThreadIdx{
	int x;
	int y;
	int z;
};

void build_shared_memory(struct BlockDim blockDim, struct BlockIdx blockIdx, struct ThreadIdx threadIdx,
		double *shared_pointer, double*global_pointer){
	int pad_height = POOL_HEIGHT/2;
	int pad_width = POOL_WIDTH/2;

	int shared_pointer_height = pad_height*2 + blockDim.y;
	int shared_pointer_width = pad_width*2 + blockDim.x;
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
}

void max_pool_kernel(double *image_pointer, double *output_pointer, struct BlockDim blockDim, struct BlockIdx blockIdx, struct ThreadIdx threadIdx)
{	

	int shared_memory_height = blockDim.y + (POOL_HEIGHT/2) * 2;
	int shared_memory_width = blockDim.x + (POOL_WIDTH/2) * 2;

	double *shared_pointer = double[shared_memory_height*shared_memory_width];

	build_shared_memory(blockDim, blockIdx, threadIdx, shared_pointer, image_pointer);
	int channel_offset = blockDim.z*HEIGHT*WIDTH;
	double max_val = 0.0;

	int i = 0;
	int j = 0;

	while(i < POOL_WIDTH){
		j = 0;
		while(j < POOL_WIDTH){
			int x_offset = threadIdx.x + j;
			int y_offset = threadIdx.y + i;
			int shared_index = y_offset*shared_memory_width + x_offset;
			double temp_max = shared_pointer[shared_index];
			if(temp_max > max_val){
				max_val = temp_max;
			}
		}
	}
	int global_x_index = blockDim.x*blockIdx.x + threadIdx.x;
	int global_y_index = blockDim.y*blockIdx.y + threadIdx.y;

	int global_index = channel_offset + global_y_index*global_y_index + global_x_index;
	output_pointer[global_index] = max_val;
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
  			image_pointer[index] = i+1;
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
  		printf("Check sum is wrong. Value = %lf \n",sum);
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
void print_shared_memory(double *shared_pointer){
	int shared_mem_height = (POOL_HEIGHT/2) * 2 + TILE_WIDTH;
  	int shared_mem_width = (POOL_WIDTH/2) * 2 + TILE_HEIGHT;
  	for(int i = 0; i < shared_mem_height; i++){
  		for(int j = 0; j < shared_mem_width; j++){
  			int offset = i*shared_mem_width + j;
  			double value = shared_pointer[offset];
  			printf("%d,%0.1lf \t",offset,value);
  		}
  		printf("\n");
  	}	
}
void print_image(double *image_pointer){

	int height = HEIGHT;
  	int width = WIDTH;
  	for(int i = 0; i < height; i++){
  		for(int j = 0; j < width; j++){
  			int offset = i*width + j;
  			double value = image_pointer[offset];
  			printf("%0.1lf ",value);
  		}
  		printf("\n");
  	}
}
void initialize_memory(double *pointer, int size, double value){
	for(int i = 0; i< size; i++){
		pointer[i] = value;
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
	initialize_memory(image_pointer, image_size, 0.0);
	initialize_memory(output_pointer, image_size, 0.0);
	initialize_memory(cpu_output_pointer, image_size, 0.0);
  	fill_image(CHANNELS, HEIGHT, WIDTH, image_pointer);
  	print_image(image_pointer);
  	struct BlockDim blockDim;
  	blockDim.x = 3;
  	blockDim.y = 3;

  	struct BlockIdx blockIdx;
  	blockIdx.x = 0;
  	blockIdx.y = 0;
  	blockIdx.z = 0;
  	double *shared_pointer = (double *)malloc(5*5);
  	initialize_memory(shared_pointer, 25, 0.0);
  	struct ThreadIdx threadIdx;
  	printf("\n\n");
	printf("x = %d, y = %d\n",blockIdx.x*blockDim.x,blockIdx.y*blockDim.y);

	for(int i = 0;i<3;i++){
		for(int j = 0;j<3;j++){
			threadIdx.x = j;
			threadIdx.y = i;
			build_shared_memory(blockDim, blockIdx, threadIdx, shared_pointer, image_pointer);
		}
	}
	
	print_shared_memory(shared_pointer);
	free(shared_pointer);
  	free(image_pointer);
  	free(output_pointer);
  	free(cpu_output_pointer);
  	

}



