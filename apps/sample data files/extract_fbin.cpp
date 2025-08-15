#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

void extract_top_1M(const std::string& input_file, const std::string& output_file, uint32_t num_extract = 1000000) {
    std::ifstream fin(input_file, std::ios::binary);
    if (!fin) {
        std::cerr << "Error: Cannot open input file " << input_file << std::endl;
        return;
    }

    // Read metadata (num_points and dimension)
    uint32_t num_points, dim;
    fin.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::cout << "Input file: " << input_file << "\nTotal points: " << num_points << ", Dimension: " << dim << std::endl;

    // Ensure we do not extract more points than available
    if (num_extract > num_points) {
        std::cerr << "Error: Requested " << num_extract << " points, but only " << num_points << " are available." << std::endl;
        num_extract = num_points; // Adjust to max available
    }

    // Allocate buffer for 10M points
    std::vector<float> data(num_extract * dim);

    // Read the first 10M points
    fin.read(reinterpret_cast<char*>(data.data()), num_extract * dim * sizeof(float));
    fin.close();

    // Write the extracted data to a new fbin file
    std::ofstream fout(output_file, std::ios::binary);
    if (!fout) {
        std::cerr << "Error: Cannot open output file " << output_file << std::endl;
        return;
    }

    // Write new metadata
    fout.write(reinterpret_cast<const char*>(&num_extract), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));

    // Write extracted data
    fout.write(reinterpret_cast<const char*>(data.data()), num_extract * dim * sizeof(float));
    fout.close();

    std::cout << "Successfully extracted " << num_extract << " points to " << output_file << std::endl;
}

int main() {
    std::string input_filename = "deep.350M.fbin";
    std::string output_filename = "deep_1M.fbin";
    
    extract_top_1M(input_filename, output_filename);
    
    return 0;
}
