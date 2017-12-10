#include <omp.h>  
#include <stdio.h>  
#include <iostream>
#include <string>

using namespace std;

int main()
{
	string input;

	while (true) {
		cout << "Do you want to execute another action?" << endl;
		cout << "1. Yes" << endl;
		cout << "2. No" << endl;

		cin >> input;

		if (input == "2" || input == "No") {
			break;
		}

		cout << "Please enter the path to the image";
		cin >> input;
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
