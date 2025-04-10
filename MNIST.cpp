#include "MNIST.h"

Reader::Reader(){
}

Reader::~Reader(){
}

int Reader::reverseInt (int i){
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void Reader::ReadImages(string path){
    ifstream file (path, ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int r=0;
        int c=0;

        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&r, sizeof(r));
        r = reverseInt(r);

        file.read((char*)&c, sizeof(c));
        c = reverseInt(c);

        Images.resize(number_of_images, vector<double>(r * c));
        
        for(int i=0;i<number_of_images;++i)
        {
            for(int a=0; a<r; ++a)
            {
                for(int b=0; b<c; ++b)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    Images[i][a * c + b] = (double)temp / 255.0;
                }
            }
        }
    }
}

void Reader::PrintImage(int index) const {
    if (index < 0 || index >= Images.size()){
        throw out_of_range("Index out of range");
    }

    int r = 28; 
    int c = 28;

    for (int a=0; a<r; ++a){
        for (int b=0; b<c; ++b){
            if (Images[index][a * c + b] > 0.5){
                cout << "#";
            } else{
                cout << ".";
            }
        }
        cout << endl;
    }
}

void Reader::ReadLabels(string path){
    ifstream file(path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        for(int i = 0; i < number_of_labels; i++)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));

            Labels.push_back((int)temp);
        }
    }
}

void Reader::PrintLabel(int index) const {
    if (index < 0 || index >= Labels.size()){
        throw out_of_range("Index out of range");
    }

    cout << Labels[index] << endl;
}



//* -------------------------------------- *//



void Reader::ReadTrainingImages(string path){
    ifstream file (path, ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int r=0;
        int c=0;

        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&r, sizeof(r));
        r = reverseInt(r);

        file.read((char*)&c, sizeof(c));
        c = reverseInt(c);

        TrainingImages.resize(number_of_images, vector<double>(r * c));
        
        for(int i=0;i<number_of_images;++i)
        {
            for(int a=0; a<r; ++a)
            {
                for(int b=0; b<c; ++b)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    TrainingImages[i][a * c + b] = (double)temp / 255.0;
                }
            }
        }
    }
}

void Reader::PrintTrainingImage(int index) const {
    if (index < 0 || index >= TrainingImages.size()){
        throw out_of_range("Index out of range");
    }

    int r = 28; 
    int c = 28;

    for (int a=0; a<r; ++a){
        for (int b=0; b<c; ++b){
            if (TrainingImages[index][a * c + b] > 0.5){
                cout << "#";
            } else{
                cout << ".";
            }
        }
        cout << endl;
    }
}

void Reader::ReadTrainingLabels(string path){
    ifstream file(path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        for(int i = 0; i < number_of_labels; i++)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));

            TrainingLabels.push_back((int)temp);
        }
    }
}

void Reader::PrintTrainingLabel(int index) const {
    if (index < 0 || index >= TrainingLabels.size()){
        throw out_of_range("Index out of range");
    }

    cout << TrainingLabels[index] << endl;
}

// Helper function to convert MNIST image to input vector
vector<double> Reader::imageToInput(vector<double>& image){
    return image; // Already in the right format (784 pixels normalized between 0-1)
}

// Helper function to convert label to target vector (one-hot encoding)
vector<double> Reader::labelToTarget(int label){
    vector<double> target(10, -1.0);
    target[label] = 1.0;
    return target;
}