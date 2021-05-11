#include <iostream>
// #include <jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <string>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/decision_tree/gini_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::tree;

int main(int argc, char** argv) 
{

	string path = "../data/cisco/" + string(argv[1]) + ".json";
	ifstream file(path);
	Json::Value rawJson;
	Json::Reader reader;

	reader.parse(file, rawJson);

	// cout << "JSON data\n" << rawJson << endl;
	int totalEntries = rawJson.size();
	cout << "JSON data\n" << totalEntries << endl;

	for (int entry = 0; entry < totalEntries; ++entry) {

		// cout << "Prefix: " << rawJson[entry]["prefix"] << endl;
		// cout << "Prefix length: " << rawJson[entry]["prefixLen"] << endl;
		// cout << "IP 1: " << rawJson[entry]["ip1"] << endl;
		// cout << "IP 1: " << rawJson[entry]["ip2"] << endl;
		// cout << "IP 1: " << rawJson[entry]["ip3"] << endl;
		// cout << "IP 1: " << rawJson[entry]["ip4"] << endl;
		// cout << "Next hop: " << rawJson[entry]["nextHop"] << endl;
		// cout << "Next hop as int: " << rawJson[entry]["nextHopInt"] << endl;

		// int x;
		// cin >> x;
	}

	arma::mat dataset;
	arma::Row<size_t> labels;

	string featuresPath = "../data/cisco/" + string(argv[1]) + "-features-train.csv";
	string labelsPath = "../data/cisco/" + string(argv[1]) + "-label-train.csv";
	data::Load(featuresPath, dataset);
	data::Load(labelsPath, labels);

	// dataset.raw_print("Training Data: ");
	// labels.raw_print("Labels For Training Data: ");

	// creating a tree object
	DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, double> dt;
	// (dataset, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth).
	dt.Train(dataset, labels, 2, 10, 1e-7, 3);

	// cout << "Number of children: "<< dt.NumChildren() << "\n";
	// cout << "dt.NumClasses():" << dt.NumClasses() << "\n";

	arma::mat datasetTest;
	arma::Row<size_t> labelsTest;
	arma::Row<size_t> predictions;
	
	string featuresPathTest = "../data/cisco/" + string(argv[1]) + "-features-test.csv";
	string labelsPathTest = "../data/cisco/" + string(argv[1]) + "-label-test.csv";
	data::Load(featuresPathTest, datasetTest);
	data::Load(labelsPathTest, labelsTest);
	dt.Classify(datasetTest, predictions);

	int count0 = 0;
	int count1 = 0;

	for (int i = 0; i < predictions.size(); i++) 
	{
		if (predictions(i) == labelsTest(i)) {
			count0 += 1;
		}
		else {
			count1 += 1;
		}
	}

	cout << "Count 0: " << count0 << "\n";
	cout << "Count 1: " << count1 << "\n";
	cout << "Accuracy: " << (float)count0 / (float)(count0 + count1) << "\n";

	string modelPath = "../models/" + string(argv[1]) + "-DT.bin";
	data::Save(modelPath, "decision-tree", dt, false, data::format::binary);
	
	DecisionTree<GiniGain, BestBinaryNumericSplit, AllCategoricalSplit, AllDimensionSelect, double> dt2;
	data::Load(modelPath, "decision-tree", dt2, false, data::format::binary);

	arma::Row<size_t> predictions2;
	dt2.Classify(datasetTest, predictions2);

	count0 = 0;
	count1 = 0;

	for (int i = 0; i < predictions2.size(); i++) 
	{
		if (predictions2(i) == labelsTest(i)) {
			count0 += 1;
		}
		else {
			count1 += 1;
		}
	}

	cout << "Count 0: " << count0 << "\n";
	cout << "Count 1: " << count1 << "\n";
	cout << "Accuracy for loaded model: " << (float)count0 / (float)(count0 + count1) << "\n";

	return 0;

}