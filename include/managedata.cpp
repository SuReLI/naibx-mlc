#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <cassert>
#include <numeric>  //std::iota()
// #include "managedata.hpp"

/* Reads CSV data into a 2D vector. To each CSV line a row in the multi array */
void csv_data_import(
    const std::string   & csv_file,
    std::vector<std::vector<double>> & csv_data_container,
    const char & sep,
    const bool & as_double,
    const bool & has_header)
{
    size_t count_lines_read = 0;

    /* Read external file into memory */
    std::ifstream input_csv_contents(csv_file);

    /* if model can not be loaded, interrupt */
    if (!input_csv_contents.is_open()) { // test whether opening succeeded
        std::cerr << "Input data file could not be opened.\nCHECK name of input file!" << std::endl;
        return; // exit void function
    }

    std::vector<double> result;
    std::string         new_line_of_csv;
    std::stringstream   ss_buffer;
    std::string         substr;

    while (std::getline(input_csv_contents, new_line_of_csv)) {
        if (!new_line_of_csv.empty()) {
            count_lines_read++;
            result.clear();
            substr.clear();
            ss_buffer.clear();
            ss_buffer.str(new_line_of_csv);
            std::string line = new_line_of_csv;

            /* parse csv line */
            // while( ss_buffer.good() ) {
            //     getline( ss_buffer, substr, ',' );
            //     std::cout << "(substr) = " << (substr)  << "  std::stod(substr) = " << std::stod(substr)  << std::endl;
            //     result.push_back( std::stod(substr) );
            // }

            while( ss_buffer.good() ) {
                while (ss_buffer.peek() == ' ') // skip spaces
                    ss_buffer.get();
                std::string substr;
                getline( ss_buffer, substr, ',' );
                result.push_back( std::stod(substr) );
            }
        }
        csv_data_container.push_back(result);
    }
    input_csv_contents.close();
}

double mean(std::vector<double>  vec)
{
    double sum = 0;
    if (vec.size() > 0) {
        for (size_t i = 0; i < vec.size(); i++) {
            sum += vec.at(i);
        }
        double mean = sum / (double) vec.size();
        return mean;
    }
    return 0;
}

/* BOW <int, double> */
void ImportData_arff(
    const std::string &arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::map<int, double>> &features_container,
    std::vector<std::vector<int> > &labels_container,
    std::vector<int> &observed_m,
    bool shufflerows,
    int first_valid_input_column,
    bool features_first)
{
    int counter = 0; // count number of lines read
    int line_of_feat = 0;
    int sum_of_elems = 0; // to get m_size of target vector
    int empty_lines = 0; // count empty target vectors
    features_container.clear(); labels_container.clear(); observed_m.clear();

    std::vector<int> integer_labs;
    std::map<int, double> bow;
    std::vector<std::string> result;
    std::string substr;
    std::string new_line_of_arff;
    std::stringstream ss_buffer;

    /* Read external file into memory */
    std::ifstream input_arff(arff_file);

    /* if model can not be loaded, interrupt */
    if (!input_arff.is_open()) { // test whether opening succeeded
        std::cerr << "Input data file could not be opened.\nCHECK name of input file!" << std::endl;
        return; // exit void function
    }

    /* - Iterate over lines of arff file (in buffer)
     * - Add each line to a 2D string-array
     */
    while (std::getline(input_arff, new_line_of_arff)) {
        /* Skip empty lines*/
        if (!new_line_of_arff.empty() && new_line_of_arff.size() > 2) {
            /* Skip lines starting with "@", e.g. "@data" */
            if (new_line_of_arff.find("@") == std::string::npos && new_line_of_arff.find("%") == std::string::npos) {
                counter++; // counts lines read
                integer_labs.clear();
                substr.clear(); result.clear();
                bow.clear();

                // at each iter, BoW mmap is cleared and filled in
                // at the end of loop, split features and labels. Create label vector
                /* Delete: '{' & '}' from raw string */
                new_line_of_arff.erase(
                    std::find(new_line_of_arff.begin(),
                              new_line_of_arff.end(),   '{'));
                new_line_of_arff.erase(
                    std::find(new_line_of_arff.begin(),
                              new_line_of_arff.end(),   '}'));

                /* new line (string) into buffer */
                ss_buffer.clear();
                ss_buffer.str(new_line_of_arff);
                std::vector<int> binary_labels(num_of_labels, 0);

                /* parse (csv) new line */
                while( ss_buffer.good() ) {
                    getline( ss_buffer, substr, ',' );

                    /* parse substr */
                    // ex: "355 2" --> [355, 2]
                    std::vector<std::string> temp_pair;
                    std::pair<int, double> key_value_pair;
                    std::istringstream iss(substr);

                    /* put key-value pairs in int vector [key, value] */
                    for(std::string keyval_str; iss >> keyval_str; ) {
                        temp_pair.push_back((keyval_str));
                    }

                    assert(temp_pair.size() == 2);
                    key_value_pair = std::make_pair (
                        std::stoi(temp_pair.at(0)),
                        std::stod(temp_pair.at(1)));

                    // !! indexing must start at 0:
                    int key   = ( key_value_pair.first - 1 );
                    double value = key_value_pair.second;

                    /* Now key_value_pair = [355, 2], a vector of ints
                     * I wand to use 355 as a key of map
                     */

                    /* Fill in BoW for features */
                    if (key < num_features) {
                        if (bow.find( key ) != bow.end()) {
                            bow.at(key) += value;
                        }
                        else {
                            bow.insert(std::make_pair(key, value));
                        }
                    }
                    /* Turn BoW into binary target label vector */
                    else {
                        binary_labels.at(key - num_features) = 1;
                    }
                }

                // RECODE BINARY TO INTEGER LABEL
                int this_label = 0;
                for (size_t i = 0; i < binary_labels.size(); i++) {
                    this_label = binary_labels.at(i);
                    if (this_label == 1) {
                        integer_labs.push_back(i);
                    }
                }

                labels_container.push_back(integer_labs);

                sum_of_elems = 0;
                for (int bin_label : binary_labels) sum_of_elems += bin_label;
                if (sum_of_elems == 0) {empty_lines++;}

                observed_m.push_back(sum_of_elems);

                features_container.push_back(bow);
                line_of_feat++;
            }
        }
    }

    /* SHUFFLE DATA */
    auto shuffle_bow = features_container;
    shuffle_bow.clear();
    auto shuffle_labels = labels_container;
    shuffle_labels.clear();
    auto shuffle_m = observed_m;
    shuffle_m.clear();

    if (shufflerows == true) {
        assert(features_container.size() == labels_container.size());

        std::vector<int> indexes(features_container.size());
        std::iota (std::begin(indexes), std::end(indexes), 0);
        random_shuffle(std::begin(indexes), std::end(indexes));

        for (size_t i = 0; i < indexes.size(); i++) {
            shuffle_bow.push_back(features_container.at(indexes.at(i)));
            shuffle_labels.push_back(labels_container.at(indexes.at(i)));
            shuffle_m.push_back(observed_m.at(indexes.at(i)));
        }

        labels_container.swap(shuffle_labels);
        features_container.swap(shuffle_bow);
        observed_m.swap(shuffle_m);

        std::cout << "Data rows were shuffled" << std::endl;
    }

    std::cout << "\tLines of data read = " << counter << std::endl;
    std::cout << "\tNumber of empty targets = " << empty_lines  << " (" << (double)empty_lines/counter*100 << "%)"<< std::endl;

    input_arff.close();
}

/* BOW <int, int> */
void ImportData_arff(
    const std::string &arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::map<int, int>> &features_container,
    std::vector<std::vector<int> > &labels_container,
    std::vector<int> &observed_m,
    bool shufflerows,
    int first_valid_input_column,
    bool features_first, // in the arff file, features preceed labels
    int numbering_starts_at // arff attribute list starts with 1
    )
{
    int counter = 0; // count number of lines read
    int line_of_feat = 0;
    int sum_of_elems = 0; // to get m_size of target vector
    int empty_lines = 0; // count empty target vectors
    features_container.clear(); labels_container.clear(); observed_m.clear();

    std::vector<int> integer_labs;
    std::map<int, int> bow;

    std::vector<std::string> result;
    std::string substr;
    std::string new_line_of_arff;
    std::stringstream ss_buffer;

    /* Read external file into memory */
    std::ifstream input_arff(arff_file);

    /* if model can not be loaded, interrupt */
    if (!input_arff.is_open()) { // test whether opening succeeded
        std::cerr << "Input data file could not be opened.\nCHECK name of input file!" << std::endl;
        return; // exit void function
    }

    // Read strings
    while (std::getline(input_arff, new_line_of_arff)) {
        /* Skip empty lines*/
        if (!new_line_of_arff.empty() && new_line_of_arff.size() > 2) {
            /* Skip lines starting with "@", e.g. "@data" */
            if (new_line_of_arff.find("@") == std::string::npos && new_line_of_arff.find("%") == std::string::npos) {
                counter++; // counts lines read
                integer_labs.clear();
                substr.clear(); result.clear();
                bow.clear();

                // at each iter, BoW mmap is cleared and filled in
                // at the end of loop, split features and labels. Create label vector

                /* Delete: '{' & '}' from raw string */
                new_line_of_arff.erase(
                    std::find(new_line_of_arff.begin(),
                              new_line_of_arff.end(),   '{'));
                new_line_of_arff.erase(
                    std::find(new_line_of_arff.begin(),
                              new_line_of_arff.end(),   '}'));

                /* new line (string) into buffer */
                ss_buffer.clear();
                ss_buffer.str(new_line_of_arff);
                std::vector<int> binary_labels(num_of_labels, 0);

                /* parse (csv) new line */
                while( ss_buffer.good() ) {
                    getline( ss_buffer, substr, ',' );

                    /* parse substr */
                    // ex: "355 2" --> [355, 2]
                    std::vector<int> key_value_vec;
                    std::istringstream iss(substr);

                    /* put key-value pairs in int vector [key, value] */
                    for(std::string keyval_str; iss >> keyval_str; ) {
                        key_value_vec.push_back(std::stoi(keyval_str));
                    }

                    // !! indexing must start at 0:
                    int key   = ( key_value_vec.at(0) - numbering_starts_at );
                    int value =   key_value_vec.at(1);

                    /* Now key_value_vec = [355, 2], a vector of ints
                     * I wand to use 355 as a key of map
                     */

                    if (features_first == true) {
                        // {feat1, feat2, ..., featn, lab1, ..., labk}
                        /* Fill in BoW for features */
                        if (key < num_features) {
                            if (bow.find( key ) != bow.end()) {
                                bow.at(key) += value;
                            }
                            else {
                                bow.insert(std::make_pair(key, value));
                            }
                        }
                        /* Turn BoW into binary target label vector */
                        else {
                            binary_labels.at(key - num_features) = 1;
                        }
                    }
                    else {
                        /* {lab1, ..., labk, feat1, feat2, ..., featn} */
                        /* Turn BoW into binary target label vector */
                        if (key < num_of_labels) {
                            binary_labels.at(key) = 1;
                        }
                        /* Fill in BoW for features */
                        else {
                            if (bow.find( key ) != bow.end()) {
                                bow.at(key) += value;
                            }
                            else {
                                bow.insert(std::make_pair(key, value));
                            }
                        }
                    }
                }

                // RECODE BINARY TO INTEGER LABEL
                int this_label = 0;
                for (size_t i = 0; i < binary_labels.size(); i++) {
                    this_label = binary_labels.at(i);
                    if (this_label == 1) {
                        integer_labs.push_back(i);
                    }
                }

                labels_container.push_back(integer_labs);

                sum_of_elems = 0;
                for (int bin_label : binary_labels) sum_of_elems += bin_label;
                if (sum_of_elems == 0) {empty_lines++;}

                observed_m.push_back(sum_of_elems);

                features_container.push_back(bow);
                line_of_feat++;

                // PrintVector(integer_labs);
            }
        }
    }

    /* SHUFFLE DATA */
    auto shuffle_bow = features_container;
    shuffle_bow.clear();
    auto shuffle_labels = labels_container;
    shuffle_labels.clear();
    auto shuffle_m = observed_m;
    shuffle_m.clear();

    if (shufflerows == true) {
        assert(features_container.size() == labels_container.size());

        std::vector<int> indexes(features_container.size());
        std::iota (std::begin(indexes), std::end(indexes), 0);
        random_shuffle(std::begin(indexes), std::end(indexes));

        for (size_t i = 0; i < indexes.size(); i++) {
            shuffle_bow.push_back(features_container.at(indexes.at(i)));
            shuffle_labels.push_back(labels_container.at(indexes.at(i)));
            shuffle_m.push_back(observed_m.at(indexes.at(i)));
        }

        labels_container.swap(shuffle_labels);
        features_container.swap(shuffle_bow);
        observed_m.swap(shuffle_m);

        std::cout << "Data rows were shuffled" << std::endl;
    }

    std::cout << "\t    Lines of input read = " << counter << std::endl;
    std::cout << "\tNumber of empty targets = " << empty_lines  << " (" << (double)empty_lines/counter*100 << "%)"<< std::endl;

    input_arff.close();
}

/* <double> features */
void ImportData_arff(
    const std::string &arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::vector<double> > &features_container,
    std::vector<std::vector<int> > &labels_container,
    std::vector<int> &observed_m,
    bool shufflerows,
    int first_valid_input_column,
    bool features_first,
    int numbering_starts_at)
{
    int counter = 0; // count number of lines read
    int sum_of_elems = 0; // to get m_size of target vector
    int empty_lines = 0; // count empty target vectors
    features_container.clear(); labels_container.clear(); observed_m.clear();

    std::vector<double> temp_feat;
    std::vector<int> integer_labs;
    std::vector<int> binary_labels;

    /* Read external file into memory */
    std::ifstream input_arff(arff_file);

    /* if model can not be loaded, interrupt */
    if (!input_arff.is_open()) { // test whether opening succeeded
        std::cerr << "Input data file could not be opened.\nCHECK name of input file!" << std::endl;
        return; // exit void function
    }

    std::vector<std::string> result;
    std::string new_line_of_arff;
    std::stringstream ss_buffer;
    while (std::getline(input_arff, new_line_of_arff)) {
        if (!new_line_of_arff.empty()) {
            /* Skip lines starting with "@", e.g. "@data" */
            if (new_line_of_arff.find("@") == std::string::npos && new_line_of_arff.find("%") == std::string::npos) {
                counter++;
                binary_labels.clear();
                integer_labs.clear();
                temp_feat.clear();
                result.clear();

                ss_buffer.clear();
                ss_buffer.str(new_line_of_arff);

                /* parse csv line */
                while( ss_buffer.good() ) {
                    while (ss_buffer.peek() == ' ') { // skip spaces
                        ss_buffer.get();
                    }
                    std::string substr;
                    getline( ss_buffer, substr, ',' );
                    result.push_back( substr );
                }

                if (features_first == true) {
                    /* IGNORE columns up to "first_valid_input_column */
                    for (int i = first_valid_input_column; i < (num_features + first_valid_input_column); i++) {
                        temp_feat.push_back(std::stod(result.at(i)));
                    }
                    features_container.push_back(temp_feat);

                    for (int i = 0; i < num_of_labels; i++) {
                        int this_label = (std::stoi(result.at(i + num_features)));
                        binary_labels.push_back(this_label);
                        // RECODE BINARY TO INTEGER LABEL
                        if (this_label == 1) {
                            integer_labs.push_back(i);
                        }
                    }

                    labels_container.push_back(integer_labs);

                    sum_of_elems = 0;
                    for (int bin_label : binary_labels) sum_of_elems += bin_label;
                    if (sum_of_elems == 0) {empty_lines++;}

                    observed_m.push_back(sum_of_elems);
                }
                else {
                    for (int i = (first_valid_input_column); i < (num_of_labels + first_valid_input_column); i++) {
                        int this_label = (std::stoi(result.at(i)));
                        binary_labels.push_back(this_label);
                        if (this_label == 1) {
                            integer_labs.push_back(i);
                        }
                    }

                    int last_column = num_features + first_valid_input_column + num_of_labels;
                    for (int i = (first_valid_input_column + num_of_labels); i < (last_column); i++) {
                        temp_feat.push_back(std::stod(result.at(i)));
                    }
                    features_container.push_back(temp_feat);


                    labels_container.push_back(integer_labs);

                    sum_of_elems = 0;
                    for (int bin_label : binary_labels) sum_of_elems += bin_label;
                    if (sum_of_elems == 0) {empty_lines++;}

                    observed_m.push_back(sum_of_elems);

                }
            }
        }
    }

    /* SHUFFLE DATA */
    auto shuffle_bow = features_container;
    shuffle_bow.clear();
    auto shuffle_labels = labels_container;
    shuffle_labels.clear();
    auto shuffle_m = observed_m;
    shuffle_m.clear();

    if (shufflerows == true) {
        assert(features_container.size() == labels_container.size());

        std::vector<int> indexes(features_container.size());
        std::iota (std::begin(indexes), std::end(indexes), 0);
        random_shuffle(std::begin(indexes), std::end(indexes));

        for (size_t i = 0; i < indexes.size(); i++) {
            shuffle_bow.push_back(features_container.at(indexes.at(i)));
            shuffle_labels.push_back(labels_container.at(indexes.at(i)));
            shuffle_m.push_back(observed_m.at(indexes.at(i)));
        }

        labels_container.swap(shuffle_labels);
        features_container.swap(shuffle_bow);
        observed_m.swap(shuffle_m);

        std::cout << "Data rows were shuffled" << std::endl;
    }

    std::cout << "Lines of data read = " << counter << std::endl;
    std::cout << "\nNumber of empty targets = " << empty_lines  << " (" << (double)empty_lines/counter*100 << "%)"<< std::endl;
    input_arff.close();
}

/* Continuos Features in R^d */
void ImportData_nbx(
    int num_features,
    std::vector<std::vector<double> > &features,// 2D vector to store explanatory variables
    std::vector<std::vector<int> > &labels,     // 2D vector to store labels
    std::vector<std::string> &raw_data,         // raw data: all inputs as a string
    std::vector<double> &temp_features_observed,// store one row, i.e. one observation (sample)
    std::vector<int> &temp_labels_observed,      // store one row, i.e. one observation (sample)
    std::vector<int> &observed_m
    )
{
    features.clear();
    labels.clear();
    raw_data.clear();
    temp_features_observed.clear();
    temp_labels_observed.clear();
    observed_m.clear();

    int lines_read = 0;
    std::string input_line;

    // This is one of the weakest points of the implementation.
    // Must be made better
    while (std::cin >> input_line) {
        lines_read++;
        raw_data.push_back(input_line);
    }

    // Structure data into vectors
    int mm;
    int pastindex = 0;      // if == 0, no lines have been read yet
    int m_tot = 0;          // number of labels read. Init at 0, i.e. no labels read
    int current_line = 0;   // current line. Init at first line = 0
    std::vector<int>::iterator max_temp; // used in algo to get the max of a vector

    // populate containers
    while (pastindex < lines_read) {
        temp_features_observed.clear();
        temp_labels_observed.clear();

        mm = stoi(raw_data[pastindex]); // get num of labels to be added to target vector
        observed_m.push_back(mm); // store real m for each sample

        // populate vectors of explanatory variables
        //
        // WARNING: j = 1 => we are ignoring first element of every row, it is m_j. GOOD LIKE THIS
        for (int j = 1; j <= num_features; j++) {
            temp_features_observed.push_back(stod(raw_data[pastindex + j]));
        }
        features.push_back(temp_features_observed);

        // populate vectors of labels. Length NOT FIXED.
        for (int i = 1; i <= mm; i++) {
            temp_labels_observed.push_back(stoi(raw_data[pastindex + num_features + i]));
        }
        labels.push_back(temp_labels_observed);

        m_tot += mm;     // labels read so far
        current_line++; // lines read so far
        pastindex = current_line * (1 + num_features) + m_tot; // index of already read data
    }
}


/* Continuos Features in R^d */
// Read external source file
void ImportData_nbx(
    std::string file_input,
    int num_features,
    std::vector<std::vector<double> > &features,// 2D vector to store explanatory variables
    std::vector<std::vector<int> > &labels,     // 2D vector to store labels
    std::vector<std::string> &raw_data,         // raw data: all inputs as a string
    std::vector<double> &temp_features_observed,// store one row, i.e. one observation (sample)
    std::vector<int> &temp_labels_observed,      // store one row, i.e. one observation (sample)
    std::vector<int> &observed_m
    )
{
    features.clear();
    labels.clear();
    raw_data.clear();
    temp_features_observed.clear();
    temp_labels_observed.clear();
    observed_m.clear();

    int lines_read = 0;
    std::string input_line;

    /* Read external file into memory */
    std::ifstream my_input_file(file_input);
    // This is one of the weakest points of the implementation.
    // Must be made better
    while (my_input_file >> input_line) {
        lines_read++;
        raw_data.push_back(input_line);
    }

    // Structure data into vectors
    int mm;
    int pastindex = 0;      // if == 0, no lines have been read yet
    int m_tot = 0;          // number of labels read. Init at 0, i.e. no labels read
    int current_line = 0;   // current line. Init at first line = 0
    std::vector<int>::iterator max_temp; // used in algo to get the max of a vector

    // populate containers
    while (pastindex < lines_read) {
        temp_features_observed.clear();
        temp_labels_observed.clear();

        mm = stoi(raw_data[pastindex]); // get num of labels to be added to target vector
        observed_m.push_back(mm); // store real m for each sample

        // populate vectors of explanatory variables
        //
        // WARNING: j = 1 => we are ignoring first element of every row, it is m_j. GOOD LIKE THIS
        for (int j = 1; j <= num_features; j++) {
            temp_features_observed.push_back(stod(raw_data[pastindex + j]));
        }
        features.push_back(temp_features_observed);

        // populate vectors of labels. Length NOT FIXED.
        for (int i = 1; i <= mm; i++) {
            temp_labels_observed.push_back(stoi(raw_data[pastindex + num_features + i]));
        }
        labels.push_back(temp_labels_observed);

        m_tot += mm;     // labels read so far
        current_line++; // lines read so far
        pastindex = current_line * (1 + num_features) + m_tot; // index of already read data
    }
}


/* Print strings */
void PrintVector(std::vector<std::string> const &vec)
{
    if (vec.size() > 0) {
        for (size_t i = 0; i < (vec.size() - 1); i++) {
            std::cout << vec.at(i) << " ";
        }
        std::cout << vec.at(vec.size() - 1) << std::endl;
    }
    else std::cout << "empty vector" << std::endl;
}

/* Print doubles */
void PrintVector(std::vector<double> const &vec)
{
    if (vec.size() > 0) {
        for (size_t i = 0; i < (vec.size() - 1); i++) {
            std::cout << vec.at(i) << " ";
        }
        std::cout << vec.at(vec.size() - 1) << std::endl;
    }
    else std::cout << "empty vector" << std::endl;
}

void PrintVector(std::vector<std::vector<double> > const &vec)
{
    if (vec.size() > 0) {
        for (size_t i = 0; i < vec.size(); i++) {
            if (vec.at(i).size() != 0) {
                for (size_t j = 0; j < (vec.size() - 1); j++) {
                    std::cout << vec.at(i).at(j) << " ";
                }
                std::cout << vec.at(i).at(vec.size() - 1) << std::endl;
            }
            else std::cout << "empty vector" << std::endl;
        }
    }
}

/* Print ints */
void PrintVector(std::vector<int> const &vec)
{
    if (vec.size() > 0) {
        for (size_t i = 0; i < (vec.size() - 1); i++) {
            std::cout << vec.at(i) << " ";
        }
        std::cout << vec.at(vec.size() - 1) << std::endl;
    }
    else std::cout << "empty vector" << std::endl;
}

void PrintVector(std::vector<std::vector<int> > const &vec)
{
    if (vec.size() > 0) {
        for (size_t i = 0; i < vec.size(); i++) {

            if (vec.at(i).size() > 0) {
                // if (vec.at(i).size() > 1) {
                //     for (size_t j = 0; j < (vec.size() - 1); j++) {
                //         std::cout << vec.at(i).at(j) << " ";
                //     }
                //     std::cout << vec.at(i).at(vec.size() - 1) << std::endl;
                // }
                //
                // else std::cout << vec.at(i).at(0) << std::endl;
                for (auto label : vec.at(i)) {
                    std::cout << label << " ";
                }
                std::cout  << std::endl;
            }
            else std::cout << "empty vector" << std::endl;
        }
    }
}

std::vector<int> KFoldIndexes(
    const size_t nrows,
    const int how_many_folds)
{
    //i-th element will contain the number of obs in (i-1)-th partition
    std::vector<int> fold_sizes(how_many_folds);
    int partition_size = (int) nrows / how_many_folds;
    int remain = nrows % how_many_folds;

    // distribute observations, result of integer division
    for (size_t i = 0; i < fold_sizes.size(); i++) {
        fold_sizes.at(i) = partition_size;
    }
    // partition remainder of division on the first partitions
    for (int i = 0; i < remain; i++) {
        fold_sizes.at(i)++;
    }

    // cumulate starting indexes
    std::vector<int> start_index_fold(how_many_folds); // as many elem as partitions
    start_index_fold.at(0) = 0; // starting point at 0-th position
    for (size_t i = 1; i < fold_sizes.size(); i++) {
        start_index_fold.at(i) = start_index_fold.at(i-1) + fold_sizes.at(i-1);
    }

    return fold_sizes;
}

// <int, int> BoW
void SplitDataForKFold(
    int testfold, // fold to be tested
    int how_many_folds, // number of folds
    // imported data
    const std::vector<std::vector<int> >   & labels_container,
    const std::vector<std::map<int, int>>  & features_container,
    const std::vector<int> & observed_m,
    // containers for CV procedure
    const std::vector<int> & fold_sizes,
    std::vector<std::map<int, int> > & training_expl,
    std::vector<std::map<int, int> > & test_expl,
    std::vector<std::vector<int> >   & training_labels,
    std::vector<std::vector<int> >   & test_labels,
    std::vector<int> & observed_m_training,
    std::vector<int> & observed_m_test)
{
    // cumulate starting indexes
    std::vector<int> start_index_fold(how_many_folds); // as many elem as partitions
    start_index_fold.at(0) = 0; // starting point at 0-th position
    for (size_t i = 1; i < fold_sizes.size(); i++) {
        start_index_fold.at(i) = start_index_fold.at(i-1) + fold_sizes.at(i-1);
    }

    if (testfold != (how_many_folds-1)) {
        std::vector<std::map<int, int> > test_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold),
            features_container.begin() + start_index_fold.at(testfold + 1));

        std::vector<std::vector<int> > test_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold),
            labels_container.begin() + start_index_fold.at(testfold + 1));

        std::vector<int> observed_m_test_TEMP(
            observed_m.begin() + start_index_fold.at(testfold),
            observed_m.begin() + start_index_fold.at(testfold + 1));

        test_expl.swap(test_expl_TEMP);
        test_labels.swap(test_labels_TEMP);
        observed_m_test.swap(observed_m_test_TEMP);
    }
    else {
        std::vector<std::map<int, int> >test_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold),
            features_container.end());

        std::vector<std::vector<int> > test_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold),
            labels_container.end());

        std::vector<int> observed_m_test_TEMP(
            observed_m.begin() + start_index_fold.at(testfold),
            observed_m.end());

        test_expl.swap(test_expl_TEMP);
        test_labels.swap(test_labels_TEMP);
        observed_m_test.swap(observed_m_test_TEMP);
    }

    /* partition: TRAIN folds */
    if (testfold == 0) { // first fold is TEST
        std::vector<std::map<int, int> > training_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold + 1),
            features_container.end());

        std::vector<std::vector<int> > training_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold + 1),
            labels_container.end());

        std::vector<int> observed_m_training_TEMP(
            observed_m.begin() + start_index_fold.at(testfold + 1),
            observed_m.end());

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
    else if (testfold == (how_many_folds-1)) { // last fold is TEST
        std::vector<std::map<int, int> > training_expl_TEMP(
            features_container.begin(),
            features_container.begin() + start_index_fold.at(testfold));

            std::vector<std::vector<int> > training_labels_TEMP(
            labels_container.begin(),
            labels_container.begin() + (start_index_fold.at(testfold)));

        std::vector<int> observed_m_training_TEMP(
            observed_m.begin(),
            observed_m.begin() + (start_index_fold.at(testfold)));

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
    else {// general case
        // FIRST part - before test partition
        std::vector<std::map<int, int> > FIRST_training_expl_TEMP(
            features_container.begin(),
            features_container.begin() + start_index_fold.at(testfold));

        std::vector<std::vector<int> > FIRST_training_labels_TEMP(
            labels_container.begin(),
            labels_container.begin() + (start_index_fold.at(testfold)));

        std::vector<int> FIRST_observed_m_training_TEMP(
            observed_m.begin(),
            observed_m.begin() + (start_index_fold.at(testfold)));


        // SECOND part - AFTER test partition
        std::vector<std::map<int, int> > SECOND_training_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold + 1),
            features_container.end());

        std::vector<std::vector<int> > SECOND_training_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold + 1),
            labels_container.end());

        std::vector<int> SECOND_observed_m_training_TEMP(
            observed_m.begin() + start_index_fold.at(testfold + 1),
            observed_m.end());

        // JOIN two parts
        std::vector<std::map<int, int> > training_expl_TEMP;
        std::vector<std::vector<int> > training_labels_TEMP;
        std::vector<int> observed_m_training_TEMP;
        // features
        training_expl_TEMP.reserve( // preallocate memory
            FIRST_training_expl_TEMP.size() + SECOND_training_expl_TEMP.size());
        training_expl_TEMP.insert(
            training_expl_TEMP.end(),
            FIRST_training_expl_TEMP.begin(),
            FIRST_training_expl_TEMP.end());
        training_expl_TEMP.insert(
            training_expl_TEMP.end(),
            SECOND_training_expl_TEMP.begin(),
            SECOND_training_expl_TEMP.end());

        // labels
        training_labels_TEMP.reserve(
            FIRST_training_labels_TEMP.size() + SECOND_training_labels_TEMP.size());

        training_labels_TEMP.insert(
            training_labels_TEMP.end(),
            FIRST_training_labels_TEMP.begin(),
            FIRST_training_labels_TEMP.end());

        training_labels_TEMP.insert(
            training_labels_TEMP.end(),
            SECOND_training_labels_TEMP.begin(),
            SECOND_training_labels_TEMP.end());

        // observed m
        observed_m_training_TEMP.reserve(
            FIRST_observed_m_training_TEMP.size() + SECOND_observed_m_training_TEMP.size());

        observed_m_training_TEMP.insert(
            observed_m_training_TEMP.end(),
            FIRST_observed_m_training_TEMP.begin(),
            FIRST_observed_m_training_TEMP.end());

        observed_m_training_TEMP.insert(
            observed_m_training_TEMP.end(),
            SECOND_observed_m_training_TEMP.begin(),
            SECOND_observed_m_training_TEMP.end());

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
}

void SplitDataForKFold(
    int testfold, // fold to be tested
    int how_many_folds, // number of folds
    // imported data
    const std::vector<std::vector<int> >   & labels_container,
    const std::vector<std::vector<double> >  & features_container,
    const std::vector<int> & observed_m,
    // containers for CV procedure
    const std::vector<int> & fold_sizes,
    std::vector<std::vector<double> > & training_expl,
    std::vector<std::vector<double> > & test_expl,
    std::vector<std::vector<int> >   & training_labels,
    std::vector<std::vector<int> >   & test_labels,
    std::vector<int> & observed_m_training,
    std::vector<int> & observed_m_test)
{
    // cumulate starting indexes
    std::vector<int> start_index_fold(how_many_folds); // as many elem as partitions
    start_index_fold.at(0) = 0; // starting point at 0-th position
    for (size_t i = 1; i < fold_sizes.size(); i++) {
        start_index_fold.at(i) = start_index_fold.at(i-1) + fold_sizes.at(i-1);
    }

    if (testfold != (how_many_folds-1)) {
        std::vector<std::vector<double> > test_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold),
            features_container.begin() + start_index_fold.at(testfold + 1));

        std::vector<std::vector<int> > test_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold),
            labels_container.begin() + start_index_fold.at(testfold + 1));

        std::vector<int> observed_m_test_TEMP(
            observed_m.begin() + start_index_fold.at(testfold),
            observed_m.begin() + start_index_fold.at(testfold + 1));

        test_expl.swap(test_expl_TEMP);
        test_labels.swap(test_labels_TEMP);
        observed_m_test.swap(observed_m_test_TEMP);
    }
    else {
        std::vector<std::vector<double> > test_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold),
            features_container.end());

        std::vector<std::vector<int> > test_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold),
            labels_container.end());

        std::vector<int> observed_m_test_TEMP(
            observed_m.begin() + start_index_fold.at(testfold),
            observed_m.end());

        test_expl.swap(test_expl_TEMP);
        test_labels.swap(test_labels_TEMP);
        observed_m_test.swap(observed_m_test_TEMP);
    }

    /* partition: TRAIN folds */
    if (testfold == 0) { // first fold is TEST
        std::vector<std::vector<double> > training_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold + 1),
            features_container.end());

        std::vector<std::vector<int> > training_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold + 1),
            labels_container.end());

        std::vector<int> observed_m_training_TEMP(
            observed_m.begin() + start_index_fold.at(testfold + 1),
            observed_m.end());

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
    else if (testfold == (how_many_folds-1)) { // last fold is TEST
        std::vector<std::vector<double> > training_expl_TEMP(
            features_container.begin(),
            features_container.begin() + start_index_fold.at(testfold));

            std::vector<std::vector<int> > training_labels_TEMP(
            labels_container.begin(),
            labels_container.begin() + (start_index_fold.at(testfold)));

        std::vector<int> observed_m_training_TEMP(
            observed_m.begin(),
            observed_m.begin() + (start_index_fold.at(testfold)));

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
    else {// general case
        // FIRST part - before test partition
        std::vector<std::vector<double> > FIRST_training_expl_TEMP(
            features_container.begin(),
            features_container.begin() + start_index_fold.at(testfold));

        std::vector<std::vector<int> > FIRST_training_labels_TEMP(
            labels_container.begin(),
            labels_container.begin() + (start_index_fold.at(testfold)));

        std::vector<int> FIRST_observed_m_training_TEMP(
            observed_m.begin(),
            observed_m.begin() + (start_index_fold.at(testfold)));


        // SECOND part - AFTER test partition
        std::vector<std::vector<double> > SECOND_training_expl_TEMP(
            features_container.begin() + start_index_fold.at(testfold + 1),
            features_container.end());

        std::vector<std::vector<int> > SECOND_training_labels_TEMP(
            labels_container.begin() + start_index_fold.at(testfold + 1),
            labels_container.end());

        std::vector<int> SECOND_observed_m_training_TEMP(
            observed_m.begin() + start_index_fold.at(testfold + 1),
            observed_m.end());

        // JOIN two parts
        std::vector<std::vector<double> > training_expl_TEMP;
        std::vector<std::vector<int> > training_labels_TEMP;
        std::vector<int> observed_m_training_TEMP;
        // features
        training_expl_TEMP.reserve( // preallocate memory
            FIRST_training_expl_TEMP.size() + SECOND_training_expl_TEMP.size());
        training_expl_TEMP.insert(
            training_expl_TEMP.end(),
            FIRST_training_expl_TEMP.begin(),
            FIRST_training_expl_TEMP.end());
        training_expl_TEMP.insert(
            training_expl_TEMP.end(),
            SECOND_training_expl_TEMP.begin(),
            SECOND_training_expl_TEMP.end());

        // labels
        training_labels_TEMP.reserve(
            FIRST_training_labels_TEMP.size() + SECOND_training_labels_TEMP.size());

        training_labels_TEMP.insert(
            training_labels_TEMP.end(),
            FIRST_training_labels_TEMP.begin(),
            FIRST_training_labels_TEMP.end());

        training_labels_TEMP.insert(
            training_labels_TEMP.end(),
            SECOND_training_labels_TEMP.begin(),
            SECOND_training_labels_TEMP.end());

        // observed m
        observed_m_training_TEMP.reserve(
            FIRST_observed_m_training_TEMP.size() + SECOND_observed_m_training_TEMP.size());

        observed_m_training_TEMP.insert(
            observed_m_training_TEMP.end(),
            FIRST_observed_m_training_TEMP.begin(),
            FIRST_observed_m_training_TEMP.end());

        observed_m_training_TEMP.insert(
            observed_m_training_TEMP.end(),
            SECOND_observed_m_training_TEMP.begin(),
            SECOND_observed_m_training_TEMP.end());

        training_expl.swap(training_expl_TEMP);
        training_labels.swap(training_labels_TEMP);
        observed_m_training.swap(observed_m_training_TEMP);
    }
}

void ComparePredictions(
    const std::vector< std::vector<int> > & pred_array,
    const std::vector< std::vector<int> > & obs_array)
{
    assert(pred_array.size() == obs_array.size());

    for (size_t i = 0; i < pred_array.size(); i++) {
        std::cout << "row(" << i << ")  obs: "; PrintVector(obs_array[i]);
        std::cout << "row(" << i << ") pred: "; PrintVector(pred_array[i]);
    }
}
