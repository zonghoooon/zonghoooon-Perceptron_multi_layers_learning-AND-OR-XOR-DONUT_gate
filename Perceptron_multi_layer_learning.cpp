//best case is (layers,nodes per layer) = (2,4) in my tests

#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <fstream>
#include <tuple>

# define TOL 0.02 //tolerance
# define C 0.05 //learning rate

class Calculate
{
private:
	float ERROR(float num);
public:
	float Sigmoid(float x);
	float Net(float input[], float weight[], int node);
	float FNet(float input[], float*** weight, int node, float last_weight[], int layer);
	float Forward(float*** weight, int node, float last_weight[], int layer, float target[]);
	float D_Forward(float*** weight, int node, float last_weight[], int layer);

};
class Gate
{
private:
	int NUM = 5;
public:
	void AND(int layer, int node);
	void OR(int layer, int node);
	void XOR(int layer, int node);
	void DONUT(int layer, int node);
	float* make_rand(float arr[], int node);
};
class Learning
{
public:
	void Backprop(float*** weight, float* l_delta, int node, int layer, float input[], int cnt, float* next_delta);
	void LEARN(float*** weight, float* last_weight, int node, int layer, float target[]);
};
int main()
{
	int layer;
	int node = 2;
	int choice;
	Gate gate;

	std::cout << "which Gate?(AND=1, OR=2, XOR=3, DONUT=4)";
	std::cin >> choice;
	std::cout << "layers:";
	std::cin >> layer;
	if (choice != 4) {
		std::cout << "nodes per layer:";
		std::cin >> node;
	}
	remove("weight.txt");
	remove("ERROR.txt");
	clock_t start, finish;
	double time;

	switch (choice) {
	case 1:
		start = clock();
		gate.AND(layer, node);
		finish = clock();
		break;
	case 2:
		start = clock();
		gate.OR(layer, node);
		finish = clock();
		break;
	case 3:
		start = clock();
		gate.XOR(layer, node);
		finish = clock();
		break;
	case 4:
		start = clock();
		gate.DONUT(layer, node);
		finish = clock();
		break;
	default:
		start = clock();
		std::cout << "wrong values.";
		finish = clock();
		break;
	}

	time = double(finish - start);
	std::cout << "running time is " << (time / CLOCKS_PER_SEC) << "sec.\n" << std::endl;
	system("pause");
}

float Calculate::ERROR(float num) { // error = (t-x)^2/2
	float result;
	result = num * num / 2;
	return result;
}
float Calculate::Sigmoid(float x) {
	float result;
	x = 0. - x;
	result = 1 / (1 + exp(x));
	return result;
}
float Calculate::Net(float input[], float weight[], int node) {
	float net = 0;
	float result;
	for (int i = 0; i < node + 1; i++) {
		net += input[i] * weight[i];
	}
	result = net;
	return result;
}
float Calculate::FNet(float input[], float*** weight, int node, float last_weight[], int layer) {
	float temp = 0;

	for (int i = 0; i < node + 1; i++) {
		temp += input[i] * last_weight[i];
	}
	float result = (float)temp;
	return result;
}
float Calculate::Forward(float*** weight, int node, float last_weight[], int layer, float target[]) {
	int num = pow(2, node);
	float* input = new float[node + 1];
	float y;
	float err = 0;

	for (int i = 0; i < num; i++) {
		int temp = i;
		for (int j = 0; j < node; j++) {
			input[j] = temp % 2;
			temp = temp / 2;
		}
		input[node] = -1;//last input is '-1'
		y = FNet(input, weight, node, last_weight, layer);
		err += ERROR(Sigmoid(target[i]) - Sigmoid(y));
	}
	err = err / num;
	delete[] input;
	return err;
}
float Calculate::D_Forward(float*** weight, int node, float last_weight[], int layer) { //input of DONUT gate is fixed
	float y;
	float err = 0;
	float train_set_x[9][2] = { {0.,0.},{0.,1.},{1.,0.},{1.,1.},{0.5,1.},{1.,0.5},{0.,0.5},{0.5,0.},{0.5,0.5} };
	float train_set_y[9] = { 0,0,0,0,0,0,0,0,1 };//target
	float input[9][3];
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 2; j++) {
			input[i][j] = train_set_x[i][j];
		}
		input[i][2] = -1;
	}

	for (int i = 0; i < 9; i++) {
		y = FNet(input[i], weight, 2, last_weight, layer);
		err += ERROR(Sigmoid(train_set_y[i]) - y);
	}
	err = err / 9;
	return err;
}

float* Gate::make_rand(float arr[], int node) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(-1, 1);


	for (int i = 0; i < node + 1; i++) {
		arr[i] = dis(gen);
	}
	return arr;
}
void Gate::AND(int layer = 0, int node = 0) {
	using namespace std;
	Calculate calculate;
	Learning learning;

	int num = pow(2, node);
	float*** weight = new float** [layer];
	float* last_weight = new float[node + 1];
	float* target = new float[num];
	float error = 0;
	std::ofstream file2("ERROR.txt", std::ios_base::out | std::ios_base::app);


	for (int i = 0; i < layer; i++)
	{
		weight[i] = new float* [node];
		for (int j = 0; j < node; j++) {
			weight[i][j] = new float[node + 1];
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			weight[i][j] = make_rand(weight[i][j], node);
		}
	}
	make_rand(last_weight, node);

	for (int i = 0; i < num; i++) {
		target[i] = 0;
	}
	target[num - 1] = 1;
	int iter = 0;

	while (true) {
		iter++;
		learning.LEARN(weight, last_weight, node, layer, target);
		error = calculate.Forward(weight, node, last_weight, layer, target);
		file2 << error << "\n";
		if (iter % NUM == 0) {
			std::cout << "count: " << iter << std::endl;
			std::cout << "error:" << error << std::endl;
		}
		if (error < TOL) {
			break;
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			for (int k = 0; k < node + 1; k++) {
				std::cout << weight[i][j][k] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	for (int h = 0; h < node + 1; h++) {
		std::cout << last_weight[h] << " ";
	}
	std::cout << "\n";
	for (int i = 0; i < layer; i++)
	{
		for (int j = 0; j < node; j++) {
			delete[] weight[i][j];
		}
	}
	for (int i = 0; i < layer; i++) {
		delete[] weight[i];
	}
	file2.close();
	delete[] weight;
	delete[] last_weight;
	delete[] target;
}
void Gate::OR(int layer = 0, int node = 0) {
	using namespace std;
	Calculate calculate;
	Learning learning;

	int num = pow(2, node);
	float*** weight = new float** [layer];
	float* last_weight = new float[node + 1];
	float* target = new float[num];
	float error = 0;
	std::ofstream file2("ERROR.txt", std::ios_base::out | std::ios_base::app);

	for (int i = 0; i < layer; i++)
	{
		weight[i] = new float* [node];
		for (int j = 0; j < node; j++) {
			weight[i][j] = new float[node + 1];
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			weight[i][j] = make_rand(weight[i][j], node);
		}
	}
	make_rand(last_weight, node);

	for (int i = 0; i < num; i++) {
		target[i] = 1;
	}
	target[0] = 0;
	int iter = 0;
	while (true) {
		iter++;
		learning.LEARN(weight, last_weight, node, layer, target);
		error = calculate.Forward(weight, node, last_weight, layer, target);
		file2 << error << "\n";
		if (iter % NUM == 0) {
			std::cout << "count: " << iter << std::endl;
			std::cout << "error:" << error << std::endl;
		}		if (error < TOL) {
			break;
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			for (int k = 0; k < node + 1; k++) {
				std::cout << weight[i][j][k] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	for (int h = 0; h < node + 1; h++) {
		std::cout << last_weight[h] << " ";
	}
	std::cout << "\n";
	for (int i = 0; i < layer; i++)
	{
		for (int j = 0; j < node; j++) {
			delete[] weight[i][j];
		}
	}
	for (int i = 0; i < layer; i++) {
		delete[] weight[i];
	}
	file2.close();
	delete[] weight;
	delete[] last_weight;
	delete[] target;
}
void Gate::XOR(int layer = 0, int node = 0) {
	using namespace std;
	Calculate calculate;
	Learning learning;

	int num = pow(2, node);
	float*** weight = new float** [layer];
	float* last_weight = new float[node + 1];
	float* target = new float[num];
	float error = 0;
	std::ofstream file2("ERROR.txt", std::ios_base::out | std::ios_base::app);

	for (int i = 0; i < layer; i++)
	{
		weight[i] = new float* [node];
		for (int j = 0; j < node; j++) {
			weight[i][j] = new float[node + 1];
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			weight[i][j] = make_rand(weight[i][j], node);
		}
	}
	make_rand(last_weight, node);

	for (int i = 1; i < num; i++) {
		target[i] = 1;
	}
	target[num - 1] = 0;
	target[0] = 0;
	int iter = 0;
	while (true) {
		iter++;
		learning.LEARN(weight, last_weight, node, layer, target);
		error = calculate.Forward(weight, node, last_weight, layer, target);
		file2 << error << "\n";
		if (iter % NUM == 0) {
			std::cout << "count: " << iter << std::endl;
			std::cout << "error:" << error << std::endl;
		}		if (error < TOL) {
			break;
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			for (int k = 0; k < node + 1; k++) {
				std::cout << weight[i][j][k] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	for (int h = 0; h < node + 1; h++) {
		std::cout << last_weight[h] << " ";
	}
	std::cout << "\n";
	for (int i = 0; i < layer; i++)
	{
		for (int j = 0; j < node; j++) {
			delete[] weight[i][j];
		}
	}
	for (int i = 0; i < layer; i++) {
		delete[] weight[i];
	}
	file2.close();
	delete[] weight;
	delete[] last_weight;
	delete[] target;
}
void Gate::DONUT(int layer = 0, int node = 0) {
	using namespace std;
	Calculate calculate;
	Learning learning;

	int num = pow(2, node);
	float*** weight = new float** [layer];
	float* last_weight = new float[node + 1];
	float target[9] = { 0,0,0,0,0,0,0,0,1 };
	float error = 0;
	std::ofstream file2("ERROR.txt", std::ios_base::out | std::ios_base::app);

	for (int i = 0; i < layer; i++)
	{
		weight[i] = new float* [node];
		for (int j = 0; j < node; j++) {
			weight[i][j] = new float[node + 1];
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			weight[i][j] = make_rand(weight[i][j], node);
		}
	}
	make_rand(last_weight, node);

	int iter = 0;
	while (true) {
		iter++;
		learning.LEARN(weight, last_weight, node, layer, target);
		error = calculate.D_Forward(weight, node, last_weight, layer);
		file2 << error << "\n";
		if (iter % NUM == 0) {
			std::cout << "count: " << iter << std::endl;
			std::cout << "error:" << error << std::endl;
		}		if (error < TOL) {
			break;
		}
	}
	for (int i = 0; i < layer; i++) {
		for (int j = 0; j < node; j++) {
			for (int k = 0; k < node + 1; k++) {
				std::cout << weight[i][j][k] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	for (int h = 0; h < node + 1; h++) {
		std::cout << last_weight[h] << " ";
	}
	std::cout << "\n";
	for (int i = 0; i < layer; i++)
	{
		for (int j = 0; j < node; j++) {
			delete[] weight[i][j];
		}
	}
	for (int i = 0; i < layer; i++) {
		delete[] weight[i];
	}
	file2.close();
	delete[] weight;
	delete[] last_weight;
}

void Learning::Backprop(float*** weight, float* l_delta, int node, int layer, float input[], int cnt, float* next_delta) {
	Calculate calc;
	float* delta = new float[node + 1];
	float temp1, temp2;
	if (cnt >= 2) {
		for (int i = 0; i < node; i++) {
			temp1 = calc.FNet(input, weight, node, weight[cnt - 1][i], cnt - 1);
			temp1 = calc.Sigmoid(temp1);
			temp2 = calc.FNet(input, weight, node, weight[cnt - 2][i], cnt - 2);
			temp2 = calc.Sigmoid(temp2);

			for (int j = 0; j < node; j++) {
				delta[j] = l_delta[j] * temp1 * (1 - temp1);
			}
			for (int k = 0; k < node; k++) {
				next_delta[i] += delta[k] * weight[cnt - 2][i][k];
			}
			for (int j = 0; j < node + 1; j++) {
				weight[cnt - 1][i][j] += -C * delta[j] * temp2;
			}
		}
	}
	else if (cnt == 1) {
		for (int i = 0; i < node; i++) {
			temp1 = calc.FNet(input, weight, node, weight[0][i], cnt - 1);
			temp1 = calc.Sigmoid(temp1);
			temp2 = calc.Sigmoid(input[i]);

			for (int j = 0; j < node + 1; j++) {
				delta[j] = l_delta[j] * temp1 * (1 - temp1);

				for (int j = 0; j < node + 1; j++) {
					weight[0][i][j] += -C * delta[j] * temp2;

				}
			}
		}
		delete[] delta;

	}
}
void Learning::LEARN(float*** weight, float* last_weight, int node, int layer, float target[]) {
	Calculate calc;

	int num = pow(2, node);
	float* input = new float[node + 1];
	float* delta = new float[node + 1];
	float* l_delta = new float[node + 1];
	float* next_delta = new float[node + 1];
	float y;
	float err = 0;
	float init_l_delta;
	float init_delta;
	float temp_x;

	std::ofstream file("weight.txt", std::ios_base::out | std::ios_base::app);

	for (int i = 0; i < num; i++) {
		int temp = i;
		for (int j = 0; j < node; j++) {
			input[j] = temp % 2;
			temp = temp / 2;
		}
		input[node] = -1;//last input is '-1'
		y = calc.FNet(input, weight, node, last_weight, layer);
		init_l_delta = -(calc.Sigmoid(target[i]) - calc.Sigmoid(y));
		init_delta = init_l_delta * (calc.Sigmoid(y)) * (1 - calc.Sigmoid(y));
		for (int j = 0; j < node; j++) {
			l_delta[j] = last_weight[j] * init_delta;
			last_weight[j] += -C * init_delta * calc.Sigmoid(calc.FNet(input, weight, node, weight[layer - 1][j], layer - 1));
		}
		last_weight[node] += -C * init_delta * (-1);
		int cnt = layer;
		for (int j = 0; j < layer; j++) {
			Backprop(weight, l_delta, node, layer, input, cnt, next_delta);
			for (int i = 0; i < node + 1; i++) {
				l_delta[i] = next_delta[i];
				next_delta[i] = 0.;
			}
			cnt = cnt - 1;
		}
	}

	for (int i = 0; i < layer; i++) {
		file << "{";
		for (int j = 0; j < node; j++) {
			file << "(";
			for (int k = 0; k < node + 1; k++) {
				file << weight[i][j][k] << ",";
			}
			file << "),";
		}
		file << "}";
	}
	file << "(";
	for (int i = 0; i < node + 1; i++) {
		file << last_weight[i] << ",";
	}
	file << ")\n";
	file.close();

	delete[] input;
	delete[] delta;
	delete[] l_delta;
	delete[] next_delta;

}
