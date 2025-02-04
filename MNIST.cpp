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

        images.resize(number_of_images, vector<double>(r * c));
        
        for(int i=0;i<number_of_images;++i)
        {
            for(int a=0; a<r; ++a)
            {
                for(int b=0; b<c; ++b)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    images[i][a * c + b] = (double)temp / 255.0;
                }
            }
        }
    }
}

void Reader::PrintImage(int index) const {
    if (index < 0 || index >= images.size()){
        throw out_of_range("Index out of range");
    }

    int r = 28; 
    int c = 28;

    for (int a=0; a<r; ++a){
        for (int b=0; b<c; ++b){
            if (images[index][a * c + b] > 0.5){
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

            labels.push_back((int)temp);
        }
    }
}

void Reader::PrintLabel(int index) const {
    if (index < 0 || index >= labels.size()){
        throw out_of_range("Index out of range");
    }

    cout << labels[index] << endl;
}