#include <omp.h>  
#include <stdio.h>  
#include <iostream>
#include <string>
#include <math.h>
#include <tuple>
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

void convert_to_grayscale(string path, string new_path)
{
	Mat image;
	image = imread(path, CV_LOAD_IMAGE_COLOR);

	int height = image.cols;
	int width = image.rows;

	Mat grayscaled_image(height, width, CV_8UC3);

	if (width && height) {
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				Vec3b clrOriginal = image.at<Vec3b>(i, j);
				double fR(clrOriginal[0]);
				double fG(clrOriginal[1]);
				double fB(clrOriginal[2]);

				float fWB = sqrt((fR * fR + fG * fG + fB * fB) / 3);

				Vec3b new_color;
				new_color[0] = static_cast<uchar>(fB);
				new_color[1] = static_cast<uchar>(fG);
				new_color[2] = static_cast<uchar>(fR);

				grayscaled_image.at<Vec3b>(i, j) = new_color;
			}
		}
	}
	save_image(grayscaled_image, new_path);
}

void resize_image(string path, string new_path) {
	Mat image;
	image = imread(path, CV_LOAD_IMAGE_COLOR);
	double horizontal_resize_percentage;
	double vertical_resize_percentage;

	cout << "Please enter horizontal resize percentage:" << endl;
	cin >> horizontal_resize_percentage;

	cout << "Please enter vertical resize percentage:" << endl;
	cin >> vertical_resize_percentage;

	int resized_rows = static_cast<int>((image.rows * vertical_resize_percentage) / 100);
	int resized_columns = static_cast<int>((image.cols * horizontal_resize_percentage) / 100);

	int initial_columns = image.cols;
	int initial_rows = image.rows;

	Size new_size(resized_columns, resized_rows);
	Mat resized_image(new_size, CV_8UC3);

#pragma omp parallel  
	{
#pragma omp for 
		for (int index = 0; index < resized_columns * resized_rows; index++) {
			int current_row = index / resized_columns;
			int current_column = index % resized_columns;

			double horizontal_projection_center = (current_column * 100) / horizontal_resize_percentage;
			double vertical_projection_center = (current_row * 100) / vertical_resize_percentage;

			vector<tuple<int, float>> unbounded_horizontal_indices_weights;
			vector<tuple<int, float>> unbounded_vertical_indices_weights;

			int left_limit = 0;
			int right_limit = 0;
			int bottom_limit = 0;
			int top_limit = 0;

			if (horizontal_resize_percentage > 100) {
				tuple<int, float> left_limit(floor(horizontal_projection_center), 1 - abs(floor(horizontal_projection_center) - horizontal_projection_center));
				tuple<int, float> right_limit(ceill(horizontal_projection_center), 1 - abs(ceill(horizontal_projection_center) - horizontal_projection_center)); 

				unbounded_horizontal_indices_weights.push_back(left_limit);
				unbounded_horizontal_indices_weights.push_back(right_limit);
			}
			else {
				double horizontal_distance = horizontal_resize_percentage / 100;

				tuple<int, float> left_limit(floor(horizontal_projection_center - horizontal_distance), 1 - abs(floor(horizontal_projection_center - horizontal_distance) - (horizontal_projection_center - horizontal_distance)));
				tuple<int, float> right_limit(ceill(horizontal_projection_center + horizontal_distance), 1 - abs(ceill(horizontal_projection_center + horizontal_distance) - (horizontal_projection_center + horizontal_distance)));

				unbounded_horizontal_indices_weights.push_back(left_limit);
				unbounded_horizontal_indices_weights.push_back(right_limit);

				for (int i = floor(horizontal_projection_center - horizontal_distance) + 1; i <= ceill(horizontal_projection_center + horizontal_distance) - 1; i++) {
					tuple<int, float> current_weight(i, 1);

					unbounded_horizontal_indices_weights.push_back(current_weight);
				}
			}

			if (vertical_resize_percentage > 100) {
				tuple<int, float> top_limit(floor(vertical_projection_center), 1 - abs(floor(vertical_projection_center) - vertical_projection_center));
				tuple<int, float> bottom_limit(ceill(vertical_projection_center), 1 - abs(ceill(vertical_projection_center) - vertical_projection_center));

				unbounded_vertical_indices_weights.push_back(top_limit);
				unbounded_vertical_indices_weights.push_back(bottom_limit);
			}
			else {
				double vertical_distance = vertical_resize_percentage / 100;

				tuple<int, float> top_limit(floor(vertical_projection_center - vertical_distance), 1 - abs(floor(vertical_projection_center - vertical_distance) - (vertical_projection_center - vertical_distance)));
				tuple<int, float> bottom_limit(ceill(vertical_projection_center + vertical_distance), 1 - abs(ceill(vertical_projection_center + vertical_distance) - (horizontal_projection_center + vertical_distance)));

				unbounded_vertical_indices_weights.push_back(top_limit);
				unbounded_vertical_indices_weights.push_back(bottom_limit);

				for (int i = floor(vertical_projection_center - vertical_distance) + 1; i <= ceill(vertical_projection_center + vertical_distance) - 1; i++) {
					tuple<int, float> current_weight(i, 1);

					unbounded_vertical_indices_weights.push_back(current_weight);
				}
			}

			vector<tuple<int, float>> horizontal_indices_weights;
			vector<tuple<int, float>> vertical_indices_weights;

			for (int index_indices = 0; index_indices < unbounded_horizontal_indices_weights.size(); index_indices++) {
				tuple<int, float> current_weight = unbounded_horizontal_indices_weights.at(index_indices);

				if (get<0>(current_weight) >= 0 && get<0>(current_weight) < initial_columns) {
					horizontal_indices_weights.push_back(current_weight);
				 }
			}

			for (int index_indices = 0; index_indices < unbounded_vertical_indices_weights.size(); index_indices++) {
				tuple<int, float> current_weight = unbounded_vertical_indices_weights.at(index_indices);

				if (get<0>(current_weight) >= 0 && get<0>(current_weight) < initial_rows) {
					vertical_indices_weights.push_back(current_weight);
				}
			}

			double total_weights = 0;

			for (int index_in_horizontal = 0; index_in_horizontal < horizontal_indices_weights.size(); index_in_horizontal++) {
				for (int index_in_vertical = 0; index_in_vertical < vertical_indices_weights.size(); index_in_vertical++) {
					tuple<int, float> horizontal_weight = horizontal_indices_weights.at(index_in_horizontal);
					tuple<int, float> vetical_weight = vertical_indices_weights.at(index_in_vertical);

					total_weights += get<1>(horizontal_weight) * get<1>(vetical_weight);
				}
			}

			double average_red = 0;
			double average_green = 0;
			double average_blue = 0;

			double normalize_delta = 255 / 2;

			for (int index_in_horizontal = 0; index_in_horizontal < horizontal_indices_weights.size(); index_in_horizontal++) {
				for (int index_in_vertical = 0; index_in_vertical < vertical_indices_weights.size(); index_in_vertical++) {
					tuple<int, float> horizontal_weight = horizontal_indices_weights.at(index_in_horizontal);
					tuple<int, float> vertical_weight = vertical_indices_weights.at(index_in_vertical);

					Vec3b colors = image.at<Vec3b>(get<0>(vertical_weight), get<0>(horizontal_weight));

					average_blue += (colors[0] - normalize_delta)  * get<1>(horizontal_weight) * get<1>(vertical_weight);

					average_green += (colors[1] - normalize_delta)  * get<1>(horizontal_weight) * get<1>(vertical_weight);

					average_red += (colors[2] - normalize_delta)  * get<1>(horizontal_weight) * get<1>(vertical_weight);
				}
			}

			average_blue = (average_blue / total_weights) + normalize_delta;
			average_green = (average_green / total_weights) + normalize_delta;
			average_red = (average_red / total_weights) + normalize_delta;

			Vec3b new_color;

			new_color[0] = static_cast<uchar>(average_blue);
			new_color[1] = static_cast<uchar>(average_green);
			new_color[2] = static_cast<uchar>(average_red);

			resized_image.at<Vec3b>(current_row, current_column) = new_color;
		}
	}

	save_image(resized_image, new_path);
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

				rotated_image.at<Vec3b>(current_column, rows - 1 - current_row) = rotated_pixel;
			}
		}

#pragma omp barrier  

		save_image(rotated_image, new_path);
	}
	else {
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

				rotated_image.at<Vec3b>(columns - current_column - 1, current_row) = rotated_pixel;
			}
		}

#pragma omp barrier  


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

