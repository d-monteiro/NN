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

    void ReadLabels(string path);
    void PrintLabel(int index) const;

    void ReadTrainingImages(string path);
    void PrintTrainingImage(int index) const;

    void ReadTrainingLabels(string path);
    void PrintTrainingLabel(int index) const;

private:
    int reverseInt (int i);

    vector<vector<double>> Images;
    vector<int> Labels;

    vector<vector<double>> TrainingImages;
    vector<int> TrainingLabels;
};