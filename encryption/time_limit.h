#include <iostream>
#include <fstream>
#include <ostream>
#include <string>


struct TimeLimit
{
    std::string name;
    int left;    
};

int encrypt(int number, int key) {
    return number ^ key;
}
 
int decrypt(int encrypted_number, int key) {
    return encrypted_number ^ key;
}
