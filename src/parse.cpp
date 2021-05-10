#include <iostream>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <string>

using namespace std;

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

		cout << "Prefix: " << rawJson[entry]["prefix"] << endl;
		cout << "Prefix length: " << rawJson[entry]["prefixLen"] << endl;
		cout << "IP 1: " << rawJson[entry]["ip1"] << endl;
		cout << "IP 1: " << rawJson[entry]["ip2"] << endl;
		cout << "IP 1: " << rawJson[entry]["ip3"] << endl;
		cout << "IP 1: " << rawJson[entry]["ip4"] << endl;
		cout << "Next hop: " << rawJson[entry]["nextHop"] << endl;
		cout << "Next hop as int: " << rawJson[entry]["nextHopInt"] << endl;

		int x;
		cin >> x;
	}

	return 0;
}