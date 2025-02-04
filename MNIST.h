#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Reader{
public:
    Reader();
    virtual ~Reader();

    void ReadImages(string path);
    void PrintImage(int index) const;


private:
    int reverseInt (int i);
    vector<vector<double>> images;
};