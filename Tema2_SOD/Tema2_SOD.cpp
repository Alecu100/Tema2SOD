#include <omp.h>  
#include <stdio.h>  
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	string input, path, operation;

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

		cout << "Please enter the operation to perform on the image:" << endl;
		cout << "1. Resize" << endl;
		cin >> operation;


		Mat image;
		image = imread(path, CV_LOAD_IMAGE_COLOR);

		if (operation == "1" || operation == "Resize") {
			resize_image(image);
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

void resize_image(Mat image) {
	float resize_percentage;

	cout << "Please enter resize percentage:" << endl;
	cin >> resize_percentage;

	int resized_rows = static_cast<int>((image.rows * resize_percentage) / 100);
	int resized_columns = static_cast<int>((image.cols * resize_percentage) / 100);


}
