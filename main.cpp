#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

// Define the NaiveBayesClassifier class
class NaiveBayesClassifier {
private:
    std::map<std::string, std::map<std::string, int>> word_counts_per_label;
    std::map<std::string, int> label_counts;
    std::map<std::string, std::map<std::string, double>> log_likelihoods;
    std::map<std::string, double> log_priors;
    std::set<std::string> vocabulary;
    int total_posts = 0;

public:
    void train(const std::vector<std::map<std::string, std::string>>& data);
    std::string predict(const std::string& content) const;
    void print_training_info() const;
    void print_classifier_parameters() const;
};

// Tokenize content into unique words
std::set<std::string> tokenize(const std::string &content) {
    std::istringstream source(content);
    std::set<std::string> tokens;
    std::string token;
    while (source >> token) {
        tokens.insert(token);
    }
    return tokens;
}

// Helper function to split a string by a delimiter into a vector
std::vector<std::string> split(const std::string &str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to read a CSV file into a vector of map entries
std::vector<std::map<std::string, std::string>> read_csv(const std::string &filename) {
    std::vector<std::map<std::string, std::string>> data;
    std::ifstream file(filename);
    std::string line;

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return data;
    }

    if (std::getline(file, line)) {
        std::vector<std::string> headers = split(line, ',');
        while (std::getline(file, line)) {
            std::map<std::string, std::string> row;
            std::vector<std::string> values = split(line, ',');
            if (values.size() != headers.size()) {
                std::cerr << "Warning: Mismatched columns in line: " << line << std::endl;
                continue; // Skip this line
            }
            for (size_t i = 0; i < headers.size(); ++i) {
                row[headers[i]] = values[i];
            }
            data.push_back(row);
        }
    } else {
        std::cerr << "Error: No header line found in file " << filename << std::endl;
    }

    return data;
}


// Implementation of the train method
void NaiveBayesClassifier::train(const std::vector<std::map<std::string, std::string>>& data) {
    // Reset counts
    total_posts = 0;
    word_counts_per_label.clear();
    label_counts.clear();
    vocabulary.clear();
    
    // Count words and labels
    for (const auto& row : data) {
        const std::string& label = row.at("tag"); // Changed from "label" to "tag"
        auto words = tokenize(row.at("content"));
        label_counts[label] += 1;
        total_posts += 1;
        for (const auto& word : words) {
            word_counts_per_label[label][word] += 1;
            vocabulary.insert(word);
        }
    }
    
    // Compute log likelihoods and priors
    for (const auto& label_pair : label_counts) {
        const std::string& label = label_pair.first;
        int label_count = label_pair.second;
        log_priors[label] = std::log(label_count) - std::log(total_posts);
        
        int total_word_count = 0;
        for (const auto& word_pair : word_counts_per_label[label]) {
            total_word_count += word_pair.second;
        }
        
        for (const auto& word : vocabulary) {
            double word_count = word_counts_per_label[label][word];
            log_likelihoods[label][word] = std::log(word_count + 1) - std::log(total_word_count + vocabulary.size());
        }
    }
}

// Implementation of the predict method
std::string NaiveBayesClassifier::predict(const std::string& content) const {
    std::set<std::string> words = tokenize(content);
    std::string best_label;
    double best_log_prob = -std::numeric_limits<double>::infinity();
    
    for (const auto& label_pair : log_priors) {
        const std::string& label = label_pair.first;
        double log_prob = label_pair.second; // Start with the log-prior
        
        for (const auto& word : words) {
            if (log_likelihoods.at(label).count(word) > 0) {
                log_prob += log_likelihoods.at(label).at(word);
            } else {
                log_prob += std::log(1.0) - std::log(total_posts + vocabulary.size());
            }
        }
        
        if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_label = label;
        }
    }
    
    return best_label;
}

// Print training information for debugging
void NaiveBayesClassifier::print_training_info() const {
    std::cout << "Total posts: " << total_posts << std::endl;
    std::cout << "Vocabulary size: " << vocabulary.size() << std::endl;
    for (const auto& label_pair : label_counts) {
        std::cout << "Label '" << label_pair.first << "' has " << label_pair.second << " instances." << std::endl;
    }
}

// Print classifier parameters for debugging
void NaiveBayesClassifier::print_classifier_parameters() const {
    for (const auto& label_pair : log_priors) {
        std::cout << "Label: " << label_pair.first << ", Log-prior: " << label_pair.second << std::endl;
        for (const auto& word_likelihood : log_likelihoods.at(label_pair.first)) {
            std::cout << "Word: " << word_likelihood.first << ", Log-likelihood: " << word_likelihood.second << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // Set floating point precision for output
    std::cout.precision(3);
    
    // Check command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " TRAIN_FILE.csv TEST_FILE.csv [--debug]" << std::endl;
        return 1;
    }
    
    std::string train_filename = argv[1];
    std::string test_filename = argv[2];
    bool debug_mode = argc > 3 && std::string(argv[3]) == "--debug";
    
    // Read and preprocess the training data
    auto training_data = read_csv(train_filename);
    
    // Create an instance of NaiveBayesClassifier
    NaiveBayesClassifier classifier;
    
    // Train the classifier with the training data
    classifier.train(training_data);
    
    // If in debug mode, print the training information
    if (debug_mode) {
        classifier.print_training_info();
        classifier.print_classifier_parameters();
    }
    
    // Read and preprocess the testing data
    auto testing_data = read_csv(test_filename);
    
    // Make predictions on the test data
    int correct_predictions = 0;
    for (const auto& row : testing_data) {
        if (row.find("tag") == row.end() || row.find("content") == row.end()) {
            std::cerr << "Error: 'tag' or 'content' key not found in row." << std::endl;
            continue; // Skip this row
        }
        std::string true_label = row.at("tag"); // Changed from "label" to "tag"
        std::string content = row.at("content");
        std::string predicted_label = classifier.predict(content);
        
        // Here you would calculate log_prob if needed for output
        double log_prob = 0.0;  // Placeholder for log-probability score
        
        // Output the prediction result
        std::cout << "correct = " << true_label << ", predicted = " << predicted_label
                  << ", log-probability score = " << log_prob << std::endl;
        std::cout << "content = " << content << "\n\n";
        
        // Update the correct predictions count
        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }
    
    // Output the performance
    std::cout << "performance: " << correct_predictions << " / "
              << testing_data.size() << " posts predicted correctly" << std::endl;
    
    return 0;
}