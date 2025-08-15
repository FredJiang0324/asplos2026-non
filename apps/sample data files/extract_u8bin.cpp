#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

void extract_u8bin(const std::string& input_file, const std::string& output_file, int num_extract, bool every_10th) {
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Error opening input file: " << input_file << std::endl;
        return;
    }

    // Read header (num_vectors and dim)
    int32_t num_vectors, dim;
    in.read(reinterpret_cast<char*>(&num_vectors), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    std::cout << "Original file: " << num_vectors << " vectors, dimension: " << dim << std::endl;

    // Determine actual extraction count
    if (every_10th) {
        num_extract = std::min(num_extract, num_vectors / 10);  // Ensure we don't extract too much
    } else {
        num_extract = std::min(num_extract, num_vectors);
    }

    std::vector<uint8_t> buffer(dim);  // Buffer for reading a single vector
    std::vector<uint8_t> extracted_data;
    extracted_data.reserve(num_extract * dim);

    if (every_10th) {
        for (int i = 0; i < num_vectors && extracted_data.size() < static_cast<size_t>(num_extract * dim); i++) {
            in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(uint8_t));
            if (i % 10 == 0) {  // Pick every 10th vector
                extracted_data.insert(extracted_data.end(), buffer.begin(), buffer.end());
            }
        }
    } else {
        for (int i = 0; i < num_extract; i++) {
            in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(uint8_t));
            extracted_data.insert(extracted_data.end(), buffer.begin(), buffer.end());
        }
    }

    in.close();

    // Write to the output file
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening output file: " << output_file << std::endl;
        return;
    }

    out.write(reinterpret_cast<const char*>(&num_extract), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(extracted_data.data()), extracted_data.size() * sizeof(uint8_t));

    out.close();
    std::cout << "Saved " << num_extract << " vectors to " << output_file << std::endl;
}

int main() {
    std::string input_file = "learn.100M.u8bin";
    std::string output_file_top1M = "sift.1M.u8bin";
    //std::string output_file_every10th = "learn.10M_every10th.u8bin";

    // Extract top 1M
    extract_u8bin(input_file, output_file_top1M, 1'000'000, false);

    // Extract every 10th vector (resulting in 10M total)
    //extract_u8bin(input_file, output_file_every10th, 10'000'000, true);

    return 0;
}
