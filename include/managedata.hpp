/** @file managedata.hpp
* managedata.hpp: Header file with functions for data management, e.g. input/output re-structuring etc.
*/
#ifndef MANAGEDATA_HPP_
#define MANAGEDATA_HPP_

#include <iostream>
#include <vector>
#include <map>
#include <string>

/* Reads CSV data into a 2D vector. To each CSV line a row in the multi array */
void csv_data_import(
    const std::string   & csv_file,
    std::vector<std::vector<double>> & csv_data_container,
    const char & sep = ',',
    const bool & as_double  = true,
    const bool & has_header = false);

double mean(std::vector<double>  vec);

/* BOW <int, double> */
void ImportData_arff(
    const std::string & arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::map<int, double>> & features_container,
    std::vector<std::vector<int> > & labels_container,
    std::vector<int> & observed_m,
    bool shufflerows = true,
    int first_valid_input_column = 0,
    bool features_first = true);

/* BOW <int, int> */
void ImportData_arff(
    const std::string & arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::map<int, int>> & features_container,
    std::vector<std::vector<int> > & labels_container,
    std::vector<int> & observed_m,
    bool shufflerows = true,
    int first_valid_input_column = 0,
    bool features_first = true, // in the arff file, features preceed labels
    int numbering_starts_at = 1 // arff attribute list starts with 1
);

/* <double> features */
void ImportData_arff(
    const std::string & arff_file,
    const int num_features,
    const int num_of_labels,
    std::vector<std::vector<double> > & features_container,
    std::vector<std::vector<int> > & labels_container,
    std::vector<int> & observed_m,
    bool shufflerows = true,
    int first_valid_input_column = 0,
    bool features_first = true,
    int numbering_starts_at = 1);

/* Continuos Features in R^d */
void ImportData_nbx(
    int num_features,
    std::vector<std::vector<double> > & features,// 2D vector to store explanatory variables
    std::vector<std::vector<int> > & labels,     // 2D vector to store labels
    std::vector<std::string> & raw_data,         // raw data: all inputs as a string
    std::vector<double> & temp_features_observed,// store one row, i.e. one observation (sample)
    std::vector<int>    & temp_labels_observed,      // store one row, i.e. one observation (sample)
    std::vector<int>    & observed_m
);

/* Continuos Features in R^d */
// Read external source file
void ImportData_nbx(
    std::string file_input,
    int num_features,
    std::vector<std::vector<double> > & features,// 2D vector to store explanatory variables
    std::vector<std::vector<int> >    & labels,     // 2D vector to store labels
    std::vector<std::string> & raw_data,         // raw data: all inputs as a string
    std::vector<double> & temp_features_observed,// store one row, i.e. one observation (sample)
    std::vector<int>    & temp_labels_observed,      // store one row, i.e. one observation (sample)
    std::vector<int>    & observed_m
);

/* Print strings */
void PrintVector(std::vector<std::string> const & vec);

/* Print values */
void PrintVector(std::vector<double> const & vec);
void PrintVector(std::vector<std::vector<double> > const & vec);
void PrintVector(std::vector<int> const & vec);
void PrintVector(std::vector<std::vector<int> > const & vec);

std::vector<int> KFoldIndexes(
    const size_t nrows,
    const int how_many_folds);

// <int, int> BoW
void SplitDataForKFold(
    int testfold, // fold to be tested
    int how_many_folds, // number of folds
    // imported data
    const std::vector<std::vector<int> >   & labels_container,
    const std::vector<std::map<int, int> > & features_container,
    const std::vector<int> & observed_m,
    // containers for CV procedure
    const std::vector<int> & fold_sizes,
    std::vector<std::map<int, int> > & training_expl,
    std::vector<std::map<int, int> > & test_expl,
    std::vector<std::vector<int> >   & training_labels,
    std::vector<std::vector<int> >   & test_labels,
    std::vector<int> & observed_m_training,
    std::vector<int> & observed_m_test);

void SplitDataForKFold(
    int testfold, // fold to be tested
    int how_many_folds, // number of folds
    // imported data
    const std::vector<std::vector<int> >    & labels_container,
    const std::vector<std::vector<double> > & features_container,
    const std::vector<int> & observed_m,
    // containers for CV procedure
    const std::vector<int> & fold_sizes,
    std::vector<std::vector<double> > & training_expl,
    std::vector<std::vector<double> > & test_expl,
    std::vector<std::vector<int> >    & training_labels,
    std::vector<std::vector<int> >    & test_labels,
    std::vector<int> & observed_m_training,
    std::vector<int> & observed_m_test);

void ComparePredictions(
    const std::vector< std::vector<int> > & pred_array,
    const std::vector< std::vector<int> > & obs_array);

#endif
