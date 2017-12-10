#include <omp.h>  
#include <stdio.h>  
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool string_ends_with(string const &fullString, string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}


void save_image(Mat image_to_save, string path) {
	if (string_ends_with(path, ".png") || string_ends_with(path, ".PNG")) {

		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(6);

		imwrite(path, image_to_save, compression_params);
	}
	else if (string_ends_with(path, ".jpg") || string_ends_with(path, ".JPG") || string_ends_with(path, ".JPEG") || string_ends_with(path, ".jpeg")) {

		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(95);

		imwrite(path, image_to_save, compression_params);
	}
}

void resize_image(string path, string new_path) {
	Mat image;
	image = imread(path, CV_LOAD_IMAGE_COLOR);
	float resize_percentage;

	cout << "Please enter resize percentage:" << endl;
	cin >> resize_percentage;

	int resized_rows = static_cast<int>((image.rows * resize_percentage) / 100);
	int resized_columns = static_cast<int>((image.cols * resize_percentage) / 100);
	Size new_size(resized_columns, resized_rows);
	Mat resized_image(new_size, CV_8UC3);

#pragma omp parallel  
	{
#pragma omp for 
		for (int index = 0; index < resized_columns * resized_rows; index++) {
			int current_row = index / resized_rows;
			int current_column = index % resized_columns;


		}
	}
}

void rotate_image(string path, string new_path) {
	string operation;
	Mat image;
	int direction;
	image = imread(path, CV_LOAD_IMAGE_COLOR);

	cout << "Please enter direction to rotate" << endl;
	cout << "1. Right" << endl;
	cout << "2. Left" << endl;

	cin >> operation;

	if (operation == "1" || operation == "Right") {
		direction = 1;
	}
	else {
		direction = -1;
	}

	Mat rotated_image(image.cols, image.rows, CV_8UC3);

	int columns = image.cols;
	int rows = image.rows;
	int total_pixels = columns * rows;

	if (direction == 1) {
#pragma omp parallel  
		{
#pragma omp for 
			for (int index = 0; index < total_pixels; index++) {
				int current_row = index / columns;
				int current_column = index % columns;

				Vec3b rotated_pixel;
				Vec3b original_pixel = image.at<Vec3b>(current_row, current_column);

				rotated_pixel[0] = original_pixel[0];
				rotated_pixel[1] = original_pixel[1];
				rotated_pixel[2] = original_pixel[2];

				rotated_image.at<Vec3b>(current_column, rows - current_row - 1) = rotated_pixel;
			}
		}

#pragma omp barrier  

		save_image(rotated_image, new_path);
	}
	else {

		save_image(rotated_image, new_path);
	}
}


int main()
{
	string input, path, operation, new_path;

	while (true) {
		cout << "Do you want to execute another action:" << endl;
		cout << "1. Yes" << endl;
		cout << "2. No" << endl;

		cin >> input;

		if (input == "2" || input == "No") {
			break;
		}

		cout << "Please enter the path to the image:" << endl;
		cin >> path;

		cout << "Please enter the path to save the new image:" << endl;
		cin >> new_path;

		cout << "Please enter the operation to perform on the image:" << endl;
		cout << "1. Resize" << endl;
		cout << "2. Rotate" << endl;
		cin >> operation;




		if (operation == "1" || operation == "Resize") {
			resize_image(path, new_path);
		}
		else if (operation == "2" || operation == "Rotate") {
			rotate_image(path, new_path);
		}
	}

	int a[5], i;
	char stop[100];

#pragma omp parallel  
	{
		// Perform some computation.  
#pragma omp for  
		for (i = 0; i < 5; i++)
			a[i] = i * i;

		// Print intermediate results.  
#pragma omp master  
		for (i = 0; i < 5; i++)
			printf_s("a[%d] = %d\n", i, a[i]);

		// Wait.  
#pragma omp barrier  

		// Continue with the computation.  
#pragma omp for  
		for (i = 0; i < 5; i++)
			a[i] += i;
	}
}

