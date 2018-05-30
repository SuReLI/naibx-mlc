/**@file: main.cpp
 *
 * @author Luca Mossina
 *
 * kfold, STARTED: 22 aug 2016
 */

#include <iostream>         // std::cin, std::cout
#include <vector>
#include <string>           // std:string, std::stod, std::stoi
#include <algorithm>        // std::max_element
#include <cstring>          //
#include <cstdlib>          // std::rand, std::srand
#include <ctime>        // std::time
#include <cassert>
#include <chrono>
#include <fstream>
#include "include/managedata.hpp"   // ImportData_nbx(), print_vector()
#include "include/naibx.hpp"
#include "include/metrics.hpp"      // HammingLoss(), Accuracy(), ...
#include "include/bownaibx.hpp"


int main(int argc, char const *argv[])
{
    /* Input parameters */
    int num_features = 0, num_of_labels = 0; //, train_samples = 0;
    int keep = 1; // obsolete

    // bool see_m = false;
    int avg_m = 0;

    bool hamming = false, accuracy = false, precision  = false;
    bool zeroone = false, recall  = false; //, f = false;
    bool all     = true; // if true, compute all loss metrics

    bool load_model = false, save_model = false;
    bool real_m = false;
    bool laplacian_smoothing = true;
    // bool print_original = false;
    bool print_prediction = false;
    bool save_output = false;

    // bool top_m = true;
    bool no_print = true;
    // bool return_sizes = false;

    std::string my_arff;
    // bool import_data_in_nbx = false;
    // bool import_data_in_arff = true;

    int kfold = 1; // if 1, no k-fold cross-validation
    double cv_split = 0.66; // default CV split ratio
    int testfold = 0; // which partition to use for testing

    std::string import_model, export_model; // names of handled models
    std::string arg; // store argv[i], user input from command line
    std::string data_format;

    bool shufflerows = true;
    bool bow_input = false;
    bool bernoulli_bow = false;
    int first_valid_data_column = 0; // ignore columns up to this one
    bool features_first = true;
    int features_numbering_starts_at = 0;

    typedef std::chrono::high_resolution_clock clock;
    double time_elapsed;
    std::chrono::time_point<clock> start_count_time;

    std::vector<double> labcard; // store folds cardinalities (estimated)

    double training_time = 0.0;
    double prediction_time = 0.0;

    /* Parse command line parameters */
    // PARSE: NO PRINT
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.find("no_print") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            if (input_value == "true") {
                no_print = true;
            }
        }
    }
    // PARSE: HELP & OPTIONAL parameters
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        // HELP requested?
        if ((arg == "-h") || (arg == "--help")) {
            std::cout << "--help [-h]: print this help message." << std::endl;;
            std::cout << "" << std::endl;
            std::cout << "dim_x:\tnumber of features in dataset" << std::endl;
            std::cout << "labels:\tnumber of labels in dataset" << std::endl;
            std::cout << "train:\tnumber of training samples" << std::endl;
            std::cout << "EXAMPLE of usage:" << std::endl;
            std::cout << argv[0] << " < ~/path/to/data dim_x=12 labels=3 train=1500 [hamming]" << std::endl;;
            return 1; // EXIT main
        }

        // PRINT list of optional parameters
        if ((arg == "-o") || (arg == "--optional")) {
            std::cout << "optional input parameters:\n" << std::endl;
            std::cout << "hamming:   return hamming loss" << std::endl;
            std::cout << "precision: ..." << std::endl;
            // TODO: add the rest
            return 1;
        }
    }

    // PARSE: model parameters: dim(feature space), number of target labels etc..
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        // Number of features, i.e. explanatory variables
        if (arg.find("format") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            data_format = (input_value);
        }
        // data path
        if (arg.find("data") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            my_arff = (input_value);
            std::cout << "Input file: " << my_arff << std::endl;
        }

        // Number of features, i.e. explanatory variables
        if (arg.find("dim_x") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            num_features = stoi(input_value);
        }
        if (arg.find("--bow") != std::string::npos) {
            bow_input = true; // load model!!
            std::cout << " --- Features are (multinomial) Bag-of-Words" << import_model << std::endl;
        }
        if (arg.find("--bernoulli") != std::string::npos) {
            bernoulli_bow = true; // load model!!
            std::cout << " --- Features are (bernoulli) Bag-of-Words" << import_model << std::endl;
        }
        if (arg.find("--no_shuffle") != std::string::npos) {
            shufflerows = false; // load model!!
            std::cout << " --- Entry row were NOT shuffled." << import_model << std::endl;
        }
        if (arg.find("skip") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            first_valid_data_column = stoi(input_value);
            std::cout << "first_valid_data_column = " << first_valid_data_column << std::endl;
        }

        // Num of labels. if Y = {0, 1, ... K} => we have K+1 labels
        if (arg.find("labels") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            num_of_labels = stoi(input_value);
        }
        if (arg.find("-lab_first") != std::string::npos) {
            //std::size_t pos = arg.find("=");
            //std::string input_value = arg.substr(pos + 1);
            std::cout << "labels preceed features" << std::endl;
            features_first = false;
        }
        if (arg.find("keep") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);
            keep = stoi(input_value);
        }
        if (arg.find("features_numbering_starts_at") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);
            features_numbering_starts_at = stoi(input_value);
        }

        // cross-validation split ratio
        if (arg.find("cv_split") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);
            cv_split = stof(input_value);
            if (!no_print) {
                std::cout << "CV on " << cv_split*100 << "% of data" << std::endl;
            }
        }

        // k-fold cross-validation. HOW MANY FOLDS:
        if (arg.find("kfold") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);
            kfold = stoi(input_value);
            if (!no_print) {
                std::cout << "kfold = " << kfold << std::endl;
            }
        }

        // TEMP HACK: specify which fold to use as TEST. specified by hand, to be reimplemented
        // if (arg.find("testfold") != std::string::npos) {
        //     std::size_t pos = arg.find("=");
        //     std::string input_value = arg.substr(pos + 1);
        //     testfold = (stoi(input_value) - 1);
        //
        //     // user must give a value in range: {1, 2, ..., K}, where K=numfolds
        //     assert(testfold >= 0 && testfold <= (kfold-1) && "specify TEST fold in range [1...kfold]");
        //     if (no_print) {
        //         std::cout << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")" << std::endl;
        //     }
        // }

        // /* top_m m */
        // if (arg.find("top_m") != std::string::npos) {
        //     std::size_t pos = arg.find("=");
        //     std::string input_value = arg.substr(pos + 1);
        //
        //     if (input_value == "true") {
        //         top_m = true;
        //         if (!no_print) {
        //             std::cout << "top_m:\tTRUE" << std::endl;
        //         }
        //     }
        // }
        // This option will prevent the algo from estimating "m", the length of the target vector
        //
        if (arg.find("real_m") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            if (input_value == "true") {
                real_m = true;
                if (!no_print) {
                    std::cout << "real_m:\tTRUE" << std::endl;
                }
            }
        }
        // if (arg.find("see_m") != std::string::npos) {
        //     std::size_t pos = arg.find("=");
        //     std::string input_value = arg.substr(pos + 1);
        //     if (input_value == "true") {
        //         see_m = true;
        //         if (!no_print) {
        //             std::cout << "real_m:\tTRUE" << std::endl;
        //         }
        //     }
        // }

        // specify by hand the size of target vectors
        if (arg.find("avg_m") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);
            avg_m = stoi(input_value);
            if (!no_print) {
                std::cout << "avg_m = " << avg_m << std::endl;
            }
        }

        // if (arg.find("print_original") != std::string::npos) {
        //     std::size_t pos = arg.find("=");
        //     std::string input_value = arg.substr(pos + 1);
        //
        //     if (input_value == "true") {
        //         print_original = true;
        //         if (!no_print) {
        //             std::cout << "print_original:\tTRUE" << std::endl;
        //         }
        //     }
        // }
        if (arg.find("print_prediction") != std::string::npos) {
            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            if (input_value == "true") {
                print_prediction = true;
                std::cout << "print_prediction:\tTRUE" << std::endl;
            }
        }
        // if (arg.find("return_sizes") != std::string::npos) {
        //     std::size_t pos = arg.find("=");
        //     std::string input_value = arg.substr(pos + 1);
        //
        //     if (input_value == "true") {
        //         return_sizes = true;
        //         if (!no_print) {
        //             std::cout << "return_sizes:\tTRUE" << std::endl;
        //         }
        //     }
        // }

        if (arg.find("load") != std::string::npos) {
            load_model = true; // load model!!

            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            import_model = input_value;
            std::cout << "imported model: " << import_model << std::endl;
        }

        if (arg.find("save_model") != std::string::npos) {
            save_model = true; // save_model model!!

            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            export_model = input_value;
            std::cout << "export model as: " << export_model << std::endl;
        }

        if (arg.find("save_output") != std::string::npos) {
            save_output = true; // save_output model!!

            std::size_t pos = arg.find("=");
            std::string input_value = arg.substr(pos + 1);

            export_model = input_value;
            std::cout << "saving output in: " << export_model << std::endl;
        }
    }

    // PARSE: optional arguments for loss metrics
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "all") {
            all = true;
            break; // all metrics will be calculated
        }
        else if (arg == "hamming") {
            hamming = true;
        }
        else if (arg == "zeroone") {
            zeroone = true;
        }
        else if (arg == "accuracy") {
            accuracy = true;
        }
        else if (arg == "recall") {
            recall = true;
        }
        else if (arg == "precision") {
            precision = true;
        }
        // else if (arg == "f") {
        //     f = true;
        // }
    }


    /* import input_raw_data */

    // init CONTAINERS
    std::vector<std::vector<double> > cont_feat_array; // 2D vector to store explanatory variables
    std::vector<std::map<int, int>>  count_features_bow;    // BOW features container
    std::vector<std::map<int, double>>  double_features_bow;    // BOW features container
    std::vector<std::vector<int> > labels_2D_array;     // 2D vector to store labels
    std::vector<std::string> input_raw_data;    // container of raw data: all inputs as a string

    std::vector<double> temp_features;          // store one row, i.e. one observation (sample)
    std::vector<int> temp_labels;               // store one row, i.e. one observation (sample)
    std::vector<int> observed_m;                // store observed "m", to compare with predicted "m"

    // Populate data containers

    if (bow_input || bernoulli_bow){
        ImportData_arff(
            my_arff,
            num_features, num_of_labels,
            count_features_bow, labels_2D_array, observed_m,
            shufflerows,
            first_valid_data_column,
            features_first,
            features_numbering_starts_at);
        }
    else {
        ImportData_arff(
            my_arff,
            num_features, num_of_labels,
            cont_feat_array, labels_2D_array, observed_m,
            shufflerows,
            first_valid_data_column,
            features_first);
    }

    /* Observed Label Cardinality */
    int how_many_labs = 0;
    for (int size : observed_m) how_many_labs += size;
    int numobs = observed_m.size();
    std::cout << "Label cardinality (OBSERVED) whole dataset  = " << (double) how_many_labs/numobs << std::endl;

    int lenfeat;
    size_t nrows;

    if (bow_input || bernoulli_bow) {
        lenfeat = count_features_bow.size();
        nrows = count_features_bow.size(); // nrow(dataset)
    }
    else {
        lenfeat = cont_feat_array.size();
        nrows = cont_feat_array.size(); // nrow(dataset)
    }

    int lenlab = labels_2D_array.size(); // nrow(dataset)
    assert(lenfeat == lenlab && "problems with imported data. nrow features != nrow labels");

    /* LOSS MEASURE CONTAINERS */
    std::vector<double> hl_vec; std::vector<double> zo_vec;
    std::vector<double> ac_vec; std::vector<double> re_vec;
    std::vector<double> pr_vec;

    /*
     * Data-partition: SPLIT cross-validation
     */
    if (save_model) {
        cv_split = 1; // use all data to train model
        std::cout << "all data used to TRAIN model" << std::endl;
    }
    if (load_model) {
        cv_split = 0; // use all data to train model
        std::cout << "all data used to PREDICT model" << std::endl;
    }

    if (bernoulli_bow) {
        // std::cout << "ERROR: Bernoulli implementation is broken, use --bow" << std::endl;
        // return 666;
        // containers for CV procedure
        std::vector<std::map<int, int> > training_expl;
        std::vector<std::map<int, int> > test_features;
        std::vector<std::vector<int> > training_labels;
        std::vector<std::vector<int> > test_labels;
        std::vector<int> observed_m_training;
        std::vector<int> observed_m_test;
        if (kfold == 1) {

            /* TODO: implement function */

            /* SPLIT: TRAINING vs TESTING */
            // compute number of rows to be used in training
            int const train_size = (int) count_features_bow.size() * cv_split;

            // partition features
            std::vector<std::map<int, int> > training_expl_TEMP(count_features_bow.begin(), count_features_bow.begin() + train_size);
            std::vector<std::map<int, int> > test_features_TEMP(count_features_bow.begin() + train_size, count_features_bow.end());
            // partition labels
            std::vector<std::vector<int> > training_labels_TEMP(labels_2D_array.begin(), labels_2D_array.begin() + train_size);
            std::vector<std::vector<int> > test_labels_TEMP(labels_2D_array.begin() + train_size, labels_2D_array.end());

            // partition m: observed number of labels
            std::vector<int> observed_m_training_TEMP(observed_m.begin(), observed_m.begin() + train_size);
            std::vector<int> observed_m_test_TEMP(observed_m.begin() + train_size, observed_m.end());

            training_expl.swap(training_expl_TEMP);
            test_features.swap(test_features_TEMP);
            training_labels.swap(training_labels_TEMP);
            test_labels.swap(test_labels_TEMP);
            observed_m_training.swap(observed_m_training_TEMP);
            observed_m_test.swap(observed_m_test_TEMP);

            /* TRAINING model */
            assert( (training_expl.size() + test_features.size()) == nrows && "lost some observations during CV split");

            /* STEP 1: init Classifier */
            BernoulliBowNaibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

            /* STEP 2: TRAINING the predictor with user inputs */
            int learn_size = training_expl.size();

            if (load_model == true) {
                my_naibx.load_my_model(import_model);
            }
            else {
                start_count_time = clock::now();
                // train model, line by line
                for (int i = 0; i < learn_size; i++) {
                    my_naibx.add_example(count_features_bow.at(i), labels_2D_array.at(i));
                }
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                training_time += time_elapsed;
            }

            /* STEP 3: store model for future use */
            if (save_model == true) {
                my_naibx.save_model(export_model);
            }

            /* APPLICATION of Model to dataset */

            // test obs vs predicted in training set
            std::vector<std::vector<int> > predictions_on_testing;
            std::vector<std::vector<int> > top_k_predictions;
            std::vector<int> temp_pred; // store temp prediction labels[i]

            /* Prediction on TEST Partition */
            // int show_pred = 500;
            // int iter = 0;
            // const int tot_pred_steps = test_features.size();

            // typedef std::chrono::high_resolution_clock clock;
            start_count_time = clock::now();

            for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                top_k_predictions.clear();

                temp_pred.clear();

                /* Good Ol NaiBX */
                auto i = std::distance(test_features.begin(), it); // get i index
                temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);
                predictions_on_testing.push_back(temp_pred);
            }
            // std::cout << "" << std::endl;
            time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

            prediction_time += time_elapsed;

            /* OUTPUT results to file  */
            if ( save_output == true ) {
                if (real_m == true) {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
                else {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
            }

            // could be done directly inside naibx. Dumb shortcut here, re-computing m
            std::vector<int> predicted_m_on_testing;
            for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                // PrintVector(predictions_on_testing.at(i));
                // std::cout << "gino: " << predictions_on_testing.at(i).size() << std::endl;
            }

            double how_many_labs_pred = 0;
            double how_many_labs_obs = 0;
            for (int size : observed_m_test) how_many_labs_obs += size;
            for (int size : predicted_m_on_testing) how_many_labs_pred += size;

            size_t numobs_pred = predicted_m_on_testing.size();
            // size_t numobs_obs = observed_m_test.size();
            labcard.push_back( (double) how_many_labs_pred/numobs_pred);

            /* LOSS MEASURES */
            // std::cout << "LabelAccuracy:" << std::endl;
            // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

            if (hamming == true     || all == true) {
                double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                hl_vec.push_back(hloss);
                std::string metric = "hamming";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, hloss, real_m);
                }

            }
            if (zeroone == true     || all == true) {
                double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                zo_vec.push_back(loss_m);
                std::string metric = "zeroone";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }

            }
            if (accuracy == true    || all == true) {
                double loss_m = Accuracy(test_labels , predictions_on_testing);
                ac_vec.push_back(loss_m);
                std::string metric = "accuracy";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (precision == true   || all == true) {
                double loss_m = Precision(test_labels , predictions_on_testing);
                pr_vec.push_back(loss_m);
                std::string metric = "precision";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (recall == true      || all == true) {
                double loss_m = Recall(test_labels , predictions_on_testing);
                re_vec.push_back(loss_m);
                std::string metric = "recall";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
        }

        /* Data-partition: K-FOLD validation */
        if (kfold > 1) {
            std::vector<int> fold_sizes = KFoldIndexes(nrows, kfold);

            for (int i = 0; i < kfold; i++) {
            // for (size_t i = 0; i < kfold; i++) {
                testfold = i;
                std::cout << "\n   --- partition n." << (testfold + 1) << " --- " << std::endl;
                SplitDataForKFold(
                    testfold, kfold,
                    // imported data
                    labels_2D_array, count_features_bow, observed_m,
                    // containers for CV procedure
                    fold_sizes,
                    training_expl, test_features,
                    training_labels, test_labels,
                    observed_m_training, observed_m_test);

                /* TRAINING model */
                assert(training_expl.size() + test_features.size() == nrows && "lost some observations during CV split");

                /* STEP 1: init Classifier */
                BernoulliBowNaibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

                /* STEP 2: TRAINING the predictor with user inputs */
                int learn_size = training_expl.size();

                if (load_model == true) {
                    my_naibx.load_my_model(import_model);
                }
                else {
                    start_count_time = clock::now();
                    // train model, line by line
                    for (int i = 0; i < learn_size; i++) {
                        my_naibx.add_example(count_features_bow.at(i), labels_2D_array.at(i));
                    }
                    time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                    training_time += time_elapsed;
                }

                /* STEP 3: store model for future use */
                if (save_model == true) {
                    my_naibx.save_model(export_model);
                }

                /* APPLICATION of Model to dataset */
                // test obs vs predicted in training set
                std::vector<std::vector<int> > predictions_on_testing;
                std::vector<int> temp_pred; // store temp prediction labels[i]

                /* Prediction on TEST Partition */
                start_count_time = clock::now();

                for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                    temp_pred.clear();

                    /* Good Ol NaiBX */
                    auto i = std::distance(test_features.begin(), it); // get i index
                    temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);
                    predictions_on_testing.push_back(temp_pred);

                }
                // std::cout << "" << std::endl;
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();
                prediction_time += time_elapsed;

                /* OUTPUT results to file  */
                if ( save_output == true ) {
                    if (real_m == true) {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                    else {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                }

                // could be done directly inside naibx. Dumb shortcut here, re-computing m
                std::vector<int> predicted_m_on_testing;
                for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                    predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                    // PrintVector(predictions_on_testing.at(i));
                }

                if (print_prediction) {
                    ComparePredictions(predictions_on_testing, test_labels);
                }

                double how_many_labs_pred = 0;
                double how_many_labs_obs = 0;
                for (int size : observed_m_test) how_many_labs_obs += size;
                for (int size : predicted_m_on_testing) how_many_labs_pred += size;

                // std::cout << "obser m" << std::endl;PrintVector(observed_m_test);
                // std::cout << "predicted_m_on_testing" << std::endl;PrintVector(predicted_m_on_testing);


                size_t numobs_pred = predicted_m_on_testing.size();
                // size_t numobs_obs = observed_m_test.size();
                labcard.push_back( (double) how_many_labs_pred/numobs_pred);
                // std::cout << "test partition, LCard (OBSERVED)  = " << (double) how_many_labs_obs/numobs_obs << std::endl;

                /* LOSS MEASURES */
                // std::cout << "LabelAccuracy:" << std::endl;
                // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

                if (hamming == true     || all == true) {
                    double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                    hl_vec.push_back(hloss);
                    // std::cout << "Hamming   = " << hloss << std::endl;
                    std::string metric = "hamming";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, hloss, real_m);
                    // }
                }
                if (zeroone == true     || all == true) {
                    double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                    zo_vec.push_back(loss_m);
                    // std::cout << "ZeroOne   = " << loss_m << std::endl;
                    std::string metric = "zeroone";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (accuracy == true    || all == true) {
                    double loss_m = Accuracy(test_labels , predictions_on_testing);
                    ac_vec.push_back(loss_m);
                    // std::cout << "Accuracy  = " << loss_m << std::endl;
                    std::string metric = "accuracy";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (precision == true   || all == true) {
                    double loss_m = Precision(test_labels , predictions_on_testing);
                    pr_vec.push_back(loss_m);
                    // std::cout << "Precision = " << loss_m << std::endl;
                    std::string metric = "precision";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (recall == true      || all == true) {
                    double loss_m = Recall(test_labels , predictions_on_testing);
                    re_vec.push_back(loss_m);
                    // std::cout << "Recall    = " << loss_m << std::endl;
                    std::string metric = "recall";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
            }
        }
    }
    else if (bow_input) { // BoW input data, integer frequencies
        // containers for CV procedure
        std::vector<std::map<int, int> > training_expl;
        std::vector<std::map<int, int> > test_features;
        std::vector<std::vector<int> > training_labels;
        std::vector<std::vector<int> > test_labels;
        std::vector<int> observed_m_training;
        std::vector<int> observed_m_test;
        if (kfold == 1) {

            /* TODO: implement function */

            /* SPLIT: TRAINING vs TESTING */
            // compute number of rows to be used in training
            int const train_size = (int) count_features_bow.size() * cv_split;

            // partition features
            std::vector<std::map<int, int> > training_expl_TEMP(count_features_bow.begin(), count_features_bow.begin() + train_size);
            std::vector<std::map<int, int> > test_features_TEMP(count_features_bow.begin() + train_size, count_features_bow.end());
            // partition labels
            std::vector<std::vector<int> > training_labels_TEMP(labels_2D_array.begin(), labels_2D_array.begin() + train_size);
            std::vector<std::vector<int> > test_labels_TEMP(labels_2D_array.begin() + train_size, labels_2D_array.end());

            // partition m: observed number of labels
            std::vector<int> observed_m_training_TEMP(observed_m.begin(), observed_m.begin() + train_size);
            std::vector<int> observed_m_test_TEMP(observed_m.begin() + train_size, observed_m.end());

            training_expl.swap(training_expl_TEMP);
            test_features.swap(test_features_TEMP);
            training_labels.swap(training_labels_TEMP);
            test_labels.swap(test_labels_TEMP);
            observed_m_training.swap(observed_m_training_TEMP);
            observed_m_test.swap(observed_m_test_TEMP);

            /* TRAINING model */
            assert(training_expl.size() + test_features.size() == nrows && "lost some observations during CV split");

            /* STEP 1: init Classifier */
            BowNaibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

            /* STEP 2: TRAINING the predictor with user inputs */
            int learn_size = training_expl.size();

            if (load_model == true) {
                my_naibx.load_my_model(import_model);
            }
            else {
                start_count_time = clock::now();
                // train model, line by line
                for (int i = 0; i < learn_size; i++) {
                    my_naibx.add_example(count_features_bow.at(i), labels_2D_array.at(i));
                }
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                training_time += time_elapsed;
            }

            /* STEP 3: store model for future use */
            if (save_model == true) {
                my_naibx.save_model(export_model);
            }

            /* APPLICATION of Model to dataset */

            // test obs vs predicted in training set
            std::vector<std::vector<int> > predictions_on_testing;
            std::vector<std::vector<int> > top_k_predictions;
            std::vector<int> temp_pred; // store temp prediction labels[i]

            /* Prediction on TEST Partition */
            // int show_pred = 500;
            // int iter = 0;
            // const int tot_pred_steps = test_features.size();

            // typedef std::chrono::high_resolution_clock clock;
            start_count_time = clock::now();

            for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                top_k_predictions.clear();

                temp_pred.clear();

                /* Good Ol NaiBX */
                auto i = std::distance(test_features.begin(), it); // get i index
                temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);
                predictions_on_testing.push_back(temp_pred);
            }
            // std::cout << "" << std::endl;
            time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

            prediction_time += time_elapsed;

            /* OUTPUT results to file  */
            if ( save_output == true ) {
                if (real_m == true) {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
                else {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
            }

            // could be done directly inside naibx. Dumb shortcut here, re-computing m
            std::vector<int> predicted_m_on_testing;
            for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                // PrintVector(predictions_on_testing.at(i));
                // std::cout << "gino: " << predictions_on_testing.at(i).size() << std::endl;
            }

            double how_many_labs_pred = 0;
            double how_many_labs_obs = 0;
            for (int size : observed_m_test) how_many_labs_obs += size;
            for (int size : predicted_m_on_testing) how_many_labs_pred += size;

            size_t numobs_pred = predicted_m_on_testing.size();
            // size_t numobs_obs = observed_m_test.size();
            labcard.push_back( (double) how_many_labs_pred/numobs_pred);

            /* LOSS MEASURES */
            // std::cout << "LabelAccuracy:" << std::endl;
            // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

            if (hamming == true     || all == true) {
                double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                hl_vec.push_back(hloss);
                std::string metric = "hamming";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, hloss, real_m);
                }

            }
            if (zeroone == true     || all == true) {
                double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                zo_vec.push_back(loss_m);
                std::string metric = "zeroone";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }

            }
            if (accuracy == true    || all == true) {
                double loss_m = Accuracy(test_labels , predictions_on_testing);
                ac_vec.push_back(loss_m);
                std::string metric = "accuracy";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (precision == true   || all == true) {
                double loss_m = Precision(test_labels , predictions_on_testing);
                pr_vec.push_back(loss_m);
                std::string metric = "precision";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (recall == true      || all == true) {
                double loss_m = Recall(test_labels , predictions_on_testing);
                re_vec.push_back(loss_m);
                std::string metric = "recall";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
        }

        /* Data-partition: K-FOLD validation */
        if (kfold > 1) {
            const std::vector<int> fold_sizes = KFoldIndexes(nrows, kfold);

            for (int i = 0; i < kfold; i++) {
                testfold = i;
                std::cout << "\n   --- partition n." << (testfold + 1) << " --- " << std::endl;
                SplitDataForKFold(
                    testfold, kfold,
                    // imported data
                    labels_2D_array, count_features_bow, observed_m,
                    // containers for CV procedure
                    fold_sizes,
                    training_expl, test_features,
                    training_labels, test_labels,
                    observed_m_training, observed_m_test);

                /* TRAINING model */
                assert(training_expl.size() + test_features.size() == nrows && "lost some observations during CV split");

                /* STEP 1: init Classifier */
                BowNaibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

                /* STEP 2: TRAINING the predictor with user inputs */
                int learn_size = training_expl.size();

                if (load_model == true) {
                    my_naibx.load_my_model(import_model);
                }
                else {
                    start_count_time = clock::now();
                    // train model, line by line
                    for (int i = 0; i < learn_size; i++) {
                        my_naibx.add_example(count_features_bow.at(i), labels_2D_array.at(i));
                    }
                    time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                    training_time += time_elapsed;
                }

                /* STEP 3: store model for future use */
                if (save_model == true) {
                    my_naibx.save_model(export_model);
                }

                /* APPLICATION of Model to dataset */

                // test obs vs predicted in training set
                std::vector<std::vector<int> > predictions_on_testing;
                // std::vector<std::vector<int> > top_k_predictions;
                std::vector<int> temp_pred; // store temp prediction labels[i]

                /* Prediction on TEST Partition */
                // int show_pred = 500;
                // int iter = 0;
                // const int tot_pred_steps = test_features.size();

                // typedef std::chrono::high_resolution_clock clock;
                start_count_time = clock::now();

                for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                    // top_k_predictions.clear();
                    // ++iter;
                    // // if (!no_print) {
                    //     if (iter % show_pred == 0) {
                    //         std::cout << (double) iter/tot_pred_steps * 100 << "% (done)" << std::endl;
                    //     }
                    // // }

                    temp_pred.clear();

                    /* Good Ol NaiBX */
                    auto i = std::distance(test_features.begin(), it); // get i index
                    temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);
                    predictions_on_testing.push_back(temp_pred);
                }
                // std::cout << "" << std::endl;
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();
                prediction_time += time_elapsed;

                /* OUTPUT results to file  */
                if ( save_output == true ) {
                    if (real_m == true) {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                    else {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                }

                // could be done directly inside naibx. Dumb shortcut here, re-computing m
                std::vector<int> predicted_m_on_testing;
                for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                    predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                    // PrintVector(predictions_on_testing.at(i));
                    // std::cout << "gino: " << predictions_on_testing.at(i).size() << std::endl;
                }

                if (print_prediction) {
                    ComparePredictions(predictions_on_testing, test_labels);
                }

                double how_many_labs_pred = 0;
                double how_many_labs_obs = 0;
                for (int size : observed_m_test) how_many_labs_obs += size;
                for (int size : predicted_m_on_testing) how_many_labs_pred += size;

                // std::cout << "obser m" << std::endl;PrintVector(observed_m_test);
                // std::cout << "predicted_m_on_testing" << std::endl;PrintVector(predicted_m_on_testing);


                size_t numobs_pred = predicted_m_on_testing.size();
                // size_t numobs_obs = observed_m_test.size();
                labcard.push_back( (double) how_many_labs_pred/numobs_pred);
                // std::cout << "test partition, LCard (OBSERVED)  = " << (double) how_many_labs_obs/numobs_obs << std::endl;

                /* LOSS MEASURES */
                // std::cout << "LabelAccuracy:" << std::endl;
                // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

                if (hamming == true     || all == true) {
                    double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                    hl_vec.push_back(hloss);
                    // std::cout << "Hamming   = " << hloss << std::endl;
                    std::string metric = "hamming";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, hloss, real_m);
                    // }
                }
                if (zeroone == true     || all == true) {
                    double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                    zo_vec.push_back(loss_m);
                    // std::cout << "ZeroOne   = " << loss_m << std::endl;
                    std::string metric = "zeroone";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (accuracy == true    || all == true) {
                    double loss_m = Accuracy(test_labels , predictions_on_testing);
                    ac_vec.push_back(loss_m);
                    // std::cout << "Accuracy  = " << loss_m << std::endl;
                    std::string metric = "accuracy";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (precision == true   || all == true) {
                    double loss_m = Precision(test_labels , predictions_on_testing);
                    pr_vec.push_back(loss_m);
                    // std::cout << "Precision = " << loss_m << std::endl;
                    std::string metric = "precision";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (recall == true      || all == true) {
                    double loss_m = Recall(test_labels , predictions_on_testing);
                    re_vec.push_back(loss_m);
                    // std::cout << "Recall    = " << loss_m << std::endl;
                    std::string metric = "recall";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
            }
        }
    }
    else {             // Continuous input data, X in R^d
        // containers for CV procedure
        std::vector<std::vector<double> > training_expl;
        std::vector<std::vector<double> > test_features;
        std::vector<std::vector<int> > training_labels;
        std::vector<std::vector<int> > test_labels;
        std::vector<int> observed_m_training;
        std::vector<int> observed_m_test;
        if (kfold == 1) {

            /* TODO: implement function */

            /* SPLIT: TRAINING vs TESTING */
            // compute number of rows to be used in training
            int const train_size = (int) lenfeat * cv_split;

            // partition features
            std::vector<std::vector<double> > training_expl_TEMP(cont_feat_array.begin(), cont_feat_array.begin() + train_size);
            std::vector<std::vector<double> > test_features_TEMP(cont_feat_array.begin() + train_size, cont_feat_array.end());
            // partition labels
            std::vector<std::vector<int> > training_labels_TEMP(labels_2D_array.begin(), labels_2D_array.begin() + train_size);
            std::vector<std::vector<int> > test_labels_TEMP(labels_2D_array.begin() + train_size, labels_2D_array.end());

            // partition m: observed number of labels
            std::vector<int> observed_m_training_TEMP(observed_m.begin(), observed_m.begin() + train_size);
            std::vector<int> observed_m_test_TEMP(observed_m.begin() + train_size, observed_m.end());

            training_expl.swap(training_expl_TEMP);
            test_features.swap(test_features_TEMP);
            training_labels.swap(training_labels_TEMP);
            test_labels.swap(test_labels_TEMP);
            observed_m_training.swap(observed_m_training_TEMP);
            observed_m_test.swap(observed_m_test_TEMP);

            /* TRAINING model */
            assert(training_expl.size() + test_features.size() == nrows && "lost some observations during CV split");

            /* STEP 1: init Classifier */
            Naibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

            /* STEP 2: TRAINING the predictor with user inputs */
            int learn_size = training_expl.size();

            if (load_model == true) {
                my_naibx.load_my_model(import_model);
            }
            else {
                start_count_time = clock::now();
                // train model, line by line
                for (int i = 0; i < learn_size; i++) {
                    my_naibx.add_example(cont_feat_array.at(i), labels_2D_array.at(i));
                }
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                training_time += time_elapsed;
            }

            /* STEP 3: store model for future use */
            if (save_model == true) {
                my_naibx.save_model(export_model);
            }

            /* APPLICATION of Model to dataset */

            // test obs vs predicted in training set
            std::vector<std::vector<int> > predictions_on_testing;
            std::vector<std::vector<int> > top_k_predictions;
            std::vector<int> temp_pred; // store temp prediction labels[i]

            /* Prediction on TEST Partition */
            // int show_pred = 500;
            // int iter = 0;
            // const int tot_pred_steps = test_features.size();

            // typedef std::chrono::high_resolution_clock clock;
            start_count_time = clock::now();

            for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                top_k_predictions.clear();
                // ++iter;
                // // if (!no_print) {
                //     if (iter % show_pred == 0) {
                //         std::cout << (double) iter/tot_pred_steps * 100 << "% (done)" << std::endl;
                //     }
                // // }

                temp_pred.clear();

                /* Good Ol NaiBX */
                auto i = std::distance(test_features.begin(), it); // get i index
                temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);

                // if (real_m == true && avg_m != 0) {
                //     temp_pred = my_naibx.predict_y(*it, temp_pred, avg_m, real_m);
                // }
                // else if (real_m == true) {
                //     auto i = std::distance(test_features.begin(), it); // get i index
                //     temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test[i], real_m);
                // }

                predictions_on_testing.push_back(temp_pred);
            }
            // std::cout << "" << std::endl;
            time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

            prediction_time += time_elapsed;

            /* OUTPUT results to file  */
            if ( save_output == true ) {
                if (real_m == true) {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_pred_real_m.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
                else {
                    std::string output_name;
                    std::size_t pos = my_arff.find(".");

                    if (pos != std::string::npos) {
                        output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }
                    else {
                        output_name = my_arff  + "_predictions.txt" ;
                        // std::cout << "output title: " << output_name << std::endl;
                    }

                    std::ofstream my_output (output_name, std::ios_base::app);
                    if(my_output.is_open()) {
                        my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                        for (auto line: predictions_on_testing) {
                            for (auto elem : line) {
                                my_output << elem << " " ;
                            }
                            my_output << "\n";
                        }
                        my_output.close();
                    }
                }
            }

            // could be done directly inside naibx. Dumb shortcut here, re-computing m
            std::vector<int> predicted_m_on_testing;
            for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                // PrintVector(predictions_on_testing.at(i));
                // std::cout << "gino: " << predictions_on_testing.at(i).size() << std::endl;
            }

            double how_many_labs_pred = 0;
            double how_many_labs_obs = 0;
            for (int size : observed_m_test) how_many_labs_obs += size;
            for (int size : predicted_m_on_testing) how_many_labs_pred += size;

            size_t numobs_pred = predicted_m_on_testing.size();
            // size_t numobs_obs = observed_m_test.size();

            labcard.push_back( (double) how_many_labs_pred/numobs_pred);

            /* LOSS MEASURES */
            // std::cout << "LabelAccuracy:" << std::endl;
            // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

            if (hamming == true     || all == true) {
                double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                hl_vec.push_back(hloss);
                std::string metric = "hamming";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, hloss, real_m);
                }

            }
            if (zeroone == true     || all == true) {
                double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                zo_vec.push_back(loss_m);
                std::string metric = "zeroone";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }

            }
            if (accuracy == true    || all == true) {
                double loss_m = Accuracy(test_labels , predictions_on_testing);
                ac_vec.push_back(loss_m);
                std::string metric = "accuracy";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (precision == true   || all == true) {
                double loss_m = Precision(test_labels , predictions_on_testing);
                pr_vec.push_back(loss_m);
                std::string metric = "precision";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
            if (recall == true      || all == true) {
                double loss_m = Recall(test_labels , predictions_on_testing);
                re_vec.push_back(loss_m);
                std::string metric = "recall";
                if (save_output == true) {
                    ExportMetrics(my_arff, metric, loss_m, real_m);
                }
            }
        }

        /* Data-partition: K-FOLD validation */
        if (kfold > 1) {
            for (int i = 0; i < kfold; i++) {
                testfold = i;
                std::cout << "\n   --- partition n." << (testfold + 1) << " --- " << std::endl;

                /* split data */
                std::vector<int> fold_sizes = KFoldIndexes(nrows, kfold);

                SplitDataForKFold(
                    testfold, kfold,
                    // imported data
                    labels_2D_array, cont_feat_array, observed_m,
                    // containers for CV procedure
                    fold_sizes,
                    training_expl, test_features,
                    training_labels, test_labels,
                    observed_m_training, observed_m_test);

                /* TRAINING model */
                assert(training_expl.size() + test_features.size() == nrows && "lost some observations during CV split");

                /* STEP 1: init Classifier */
                Naibx my_naibx(num_features, num_of_labels, laplacian_smoothing, keep);

                /* STEP 2: TRAINING the predictor with user inputs */
                int learn_size = training_expl.size();

                if (load_model == true) {
                    my_naibx.load_my_model(import_model);
                }
                else {
                    start_count_time = clock::now();
                    // train model, line by line
                    for (int i = 0; i < learn_size; i++) {
                        my_naibx.add_example(cont_feat_array.at(i), labels_2D_array.at(i));
                    }
                    time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                    training_time += time_elapsed;
                }

                /* STEP 3: store model for future use */
                if (save_model == true) {
                    my_naibx.save_model(export_model);
                }

                /* APPLICATION of Model to dataset */

                // test obs vs predicted in training set
                std::vector<std::vector<int> > predictions_on_testing;
                std::vector<std::vector<int> > top_k_predictions;
                std::vector<int> temp_pred; // store temp prediction labels[i]

                /* Prediction on TEST Partition */
                // int show_pred = 500;
                // int iter = 0;
                // const int tot_pred_steps = test_features.size();

                // typedef std::chrono::high_resolution_clock clock;
                start_count_time = clock::now();

                for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                    top_k_predictions.clear();
                    // ++iter;
                    // // if (!no_print) {
                    //     if (iter % show_pred == 0) {
                    //         std::cout << (double) iter/tot_pred_steps * 100 << "% (done)" << std::endl;
                    //     }
                    // // }

                    temp_pred.clear();

                    /* Good Ol NaiBX */
                    auto i = std::distance(test_features.begin(), it); // get i index
                    temp_pred = my_naibx.predict_y(*it, temp_pred, observed_m_test.at(i), real_m);

                    predictions_on_testing.push_back(temp_pred);
                }
                // std::cout << "" << std::endl;
                time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start_count_time).count();

                prediction_time += time_elapsed;

                /* OUTPUT results to file  */
                if ( save_output == true ) {
                    if (real_m == true) {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_pred_real_m.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                    else {
                        std::string output_name;
                        std::size_t pos = my_arff.find(".");

                        if (pos != std::string::npos) {
                            output_name = my_arff.substr(0, pos) + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }
                        else {
                            output_name = my_arff  + "_predictions.txt" ;
                            // std::cout << "output title: " << output_name << std::endl;
                        }

                        std::ofstream my_output (output_name, std::ios_base::app);
                        if(my_output.is_open()) {
                            my_output << "testfold = " << (testfold + 1) << " (out of "<<kfold<<")\n";
                            for (auto line: predictions_on_testing) {
                                for (auto elem : line) {
                                    my_output << elem << " " ;
                                }
                                my_output << "\n";
                            }
                            my_output.close();
                        }
                    }
                }

                // could be done directly inside naibx. Dumb shortcut here, re-computing m
                std::vector<int> predicted_m_on_testing;
                for (size_t i = 0; i < predictions_on_testing.size(); i++) {
                    predicted_m_on_testing.push_back(predictions_on_testing.at(i).size());
                }

                double how_many_labs_pred = 0;
                double how_many_labs_obs = 0;
                for (int size : observed_m_test) how_many_labs_obs += size;
                for (int size : predicted_m_on_testing) how_many_labs_pred += size;

                size_t numobs_pred = predicted_m_on_testing.size();
                // size_t numobs_obs = observed_m_test.size();
                labcard.push_back( (double) how_many_labs_pred/numobs_pred);
                // std::cout << "test partition, LCard (OBSERVED)  = " << (double) how_many_labs_obs/numobs_obs << std::endl;

                /* LOSS MEASURES */
                // std::cout << "LabelAccuracy:" << std::endl;
                // LabelAccuracy(test_labels , predictions_on_testing, num_of_labels);

                if (hamming == true     || all == true) {
                    double hloss = HammingLoss(test_labels , predictions_on_testing,  num_of_labels);
                    hl_vec.push_back(hloss);
                    std::string metric = "hamming";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, hloss, real_m);
                    // }
                }
                if (zeroone == true     || all == true) {
                    double loss_m = ZeroOneLoss(test_labels , predictions_on_testing);
                    zo_vec.push_back(loss_m);
                    std::string metric = "zeroone";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (accuracy == true    || all == true) {
                    double loss_m = Accuracy(test_labels , predictions_on_testing);
                    ac_vec.push_back(loss_m);
                    std::string metric = "accuracy";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (precision == true   || all == true) {
                    double loss_m = Precision(test_labels , predictions_on_testing);
                    pr_vec.push_back(loss_m);
                    std::string metric = "precision";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
                if (recall == true      || all == true) {
                    double loss_m = Recall(test_labels , predictions_on_testing);
                    re_vec.push_back(loss_m);
                    std::string metric = "recall";
                    // if (save_output == true) {
                    //     ExportMetrics(my_arff, metric, loss_m, real_m);
                    // }
                }
            }
        }
    }

    double mn = 0;
    std::cout << "Losses:" << std::endl;

    mn = mean(hl_vec);
    std::cout << "\tHL = " << mn << std::endl ;
    mn = mean(zo_vec);
    std::cout << "\tZO = " << mn << std::endl ;
    mn = mean(ac_vec);
    std::cout << "\tAC = " << mn << std::endl ;
    mn = mean(pr_vec);
    std::cout << "\tPR = " << mn << std::endl ;
    mn = mean(re_vec);
    std::cout << "\tRE = " << mn << std::endl ;
    mn = mean(labcard);
    std::cout << "\tLC = " << mn << std::endl ;

    std::cout << "  TRAINING TIME = " << training_time * 0.001 << " (secs)" << std::endl;
    std::cout << "PREDICTION TIME = " << prediction_time *  0.001 << " (secs)" << std::endl;

    return 0;
};
