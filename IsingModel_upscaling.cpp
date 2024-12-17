#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <execution>
#include "Eigen/Dense"
#include "NeuralNetwork.h"


//Get the training data from the specfified file
std::vector<DataPoint> get_data(const std::string& filename, uint32_t lattice_size)
{
	//Each line in file consists of the list of site values	
	std::ifstream file(filename);
	//Get the number of lines data points in the file
	uint32_t n;
	file >> n;
	
	std::vector<DataPoint> vec;
	vec.reserve(n);
	
	for(uint32_t i=0;i<n;i++)
	{
		VectorXf inputs(lattice_size);
		VectorXf expected_outputs(lattice_size);	
		for(uint32_t j=0;j<lattice_size;j++)
		{
			file >> inputs(j);
			expected_outputs(j) = inputs(j);
		}
		vec.push_back({inputs, expected_outputs});	
	}
	
	return vec;
}


Eigen::VectorXf partial_upscale(const Eigen::VectorXf& input, uint32_t size)
{
	//We have the following situation after scaling
	//The in-between sites (X) need to be given a value
	//A  X1 B
	//X2 X3 X4
	//C  X5 D
			
	Eigen::MatrixXf output = Eigen::MatrixXf::Zero(size*2,size*2);
	for(uint32_t row=0;row<size;row++)
		for(uint32_t col=0;col<size;col++)
			output(2*row,2*col) = input(row*size+col);

	for(uint32_t row=0;row<size;row++){
		for(uint32_t col=0;col<size;col++){
			if(row%2==0 && col%2==1)
			{
				//X1 and X5
				output(2*row,2*col+1) = 0.5f*(output(2*row,2*col) + output(2*row,2*col+2));
			}
			else if(row%2==1 && col%2==0)
			{
				//X2 and X4
				output(2*row+1,2*col) = 0.5f*(output(2*row,2*col) + output(2*row+2,2*col));
			}
			else if(row%2==1 && col%2==1)
			{
				//X3
				output(2*row+1,2*col+1) = 0.25f*(output(2*row,2*col+1)
				                                +output(2*row+2,2*col+1)
												+output(2*row+1,2*col)
												+output(2*row+1,2*col+2));
			}
		}
	}
	
	Eigen::VectorXf output_vec(size*size);
	for(uint32_t row=0;row<size;row++)
		for(uint32_t col=0;col<size;col++)
			output_vec(row*size+col) = output(row,col);
			
	return output_vec;
}

int main()
{
	std::vector<DataPoint> training_data = get_data("autoencoder_training_data.txt",900);
	std::vector<uint32_t> layer_sizes = {900,600,500,225,500,600,900};
	NeuralNetwork autoencoder(layer_sizes);	
	autoencoder.train(training_data,1000,10,0.01);
	
	std::ofstream file("autoencoder_output_file.txt")
	std::vector<DataPoint> testing_data = get_data("autoencoder_testing_data.txt",225);
	for(DataPoint data_point: testingdata){
		Eigen::VectorXf partial_upscaled_state = partial_upscale(data_point.inputs);
		Eigen::VectorXf upscaled_state = autoencoder.calculate_outputs(partial_upscaled_state);
		file << upscaled_state << "\n\n";
	}
}