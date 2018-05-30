#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>        // std::sort(), std::find()
#include "managedata.hpp"   // PrintVector
#include <cassert>

void ExportMetrics(std::string my_arff, std::string metr, double loss_value, bool realm = false)
{
    std::string output_name;
    std::size_t pos = my_arff.find(".");

    if (realm == true) {
        if (pos != std::string::npos) {
            output_name = my_arff.substr(0, pos) + "_real_m_" + metr + ".txt" ;
        } else {
            output_name = my_arff + "_real_m_" + metr + ".txt" ;
        }

        std::ofstream my_output (output_name, std::ios_base::app);
        if (my_output.is_open()) {
            my_output << loss_value << "\n";
            my_output.close();
        }
    } else {
        if (pos != std::string::npos) {
            output_name = my_arff.substr(0, pos) + "_metr_" + metr + ".txt" ;
        } else {
            output_name = my_arff + "_" + metr + ".txt" ;
        }

        std::ofstream my_output (output_name, std::ios_base::app);
        if (my_output.is_open()) {
            my_output << loss_value << "\n";
            my_output.close();
        }
    }
}

double LCard(
    const std::vector<std::vector<int> > & target_lab)
{
    const size_t sample_size = target_lab.size();

    if (sample_size > 0) {
        int how_many_pred_labs = 0;
        for (auto vector : target_lab) {
            how_many_pred_labs += vector.size();
        }

        double LCard = (double) how_many_pred_labs / (double) sample_size;
        return LCard;
    } else {
        std::cout << "No labels were predicted" << std::endl;
        return 0;
    }
}

void LabelAccuracy(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels)
{
    assert(observations.size() == predictions.size() && "you don't have an equal amount of predictions vectors and observations");

    std::vector<int> occurences(number_of_labels, 0);
    std::vector<int> braulio(number_of_labels, 0);

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        // get "i", iteration index
        auto i = std::distance(observations.begin(), iter);

        int label;
        for (size_t j = 0; j < observations.at(i).size(); j++) {
            label = observations.at(i).at(j);

            occurences.at(observations.at(i).at(j))++;

            if ( std::find(predictions.at(i).begin(), predictions.at(i).end(), label) != predictions.at(i).end() ) {
                braulio.at(observations.at(i).at(j))++;
            }
        }
    }
    std::vector<double> acc(number_of_labels);

    for (size_t i = 0; i < acc.size(); i++) {
        // if (occurences.at(i) != 0) {
        if (true) {
            acc.at(i) = (double) braulio.at(i) / (double) occurences.at(i);
        }
    }
    std::cout << "True positives (per label):" << std::endl;
    PrintVector(acc);
}

/**
 * ZeroOneLoss implementation of 0/1 loss function, also known as "Exact Match Ratio".
 * @param observations 2D vector of label observations, each row a sample
 * @param predictions  2D vector of label predictions, each row a sample
 * @return value in [0, 1], 0 is best
 */
double ZeroOneLoss(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions)
{
    int count_good = 0; // count exact predictions
    int sample_size = observations.size();
    double error;


    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        // get "i", iteration index
        auto i = std::distance(observations.begin(), iter);

        std::sort(observations.at(i).begin(), observations.at(i).end());
        std::sort(predictions.at(i).begin(), predictions.at(i).end());

        if (observations.at(i).size() == predictions.at(i).size()) {
            if (observations.size() == 0) {
                count_good++;
            }
            else if ((observations.at(i) == predictions.at(i))) {
                count_good++;
            }
        }
    }
    error = 1 - ((double) count_good / (double) sample_size);

    return error;
}

/**
 * [HammingLoss description]
 * @param observations  2D vector of observations, each row a sample
 * @param predictions   2D vector of label predictions, each row a sample
 * @param number_of_labels
 * @return value in [0, 1], 0 is best
 */
double HammingLoss(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels)
{
    int empty_predictions  = 0;
    int empty_observations = 0;
    int empty_correct = 0;
    for (size_t i = 0; i < observations.size(); i++) {
        if (observations.at(i).size() == 0) {
            empty_observations++;
        }
        if (predictions.at(i).size() == 0) {
            empty_predictions++;
        }
        if (predictions.at(i).size() == 0 && observations.at(i).size() == 0) {
            empty_correct++;
        }
    }
    if (empty_predictions > 0 || empty_observations > 0) {
        std::cout << "  empty_obs (% total) = " << empty_observations
        << " (" << ((double) empty_observations)/observations.size()*100 << "%)"
        << std::endl;
        std::cout << " empty_pred (% total) = " << empty_predictions
        << " (" << ((double) empty_predictions)/predictions.size()*100 << "%)"
        << std::endl;
        std::cout << "     true discoveries = " << empty_correct
        << " (" << ((double) empty_correct)/ empty_predictions * 100 << "%)"
        << std::endl;
        std::cout << "    % of discoverable = " << empty_correct
        << " (" << ((double) empty_correct)/ empty_observations * 100 << "%)"
        << std::endl;
    }

    double concordance;
    double sum_of_concordances = 0;
    int sample_size = observations.size();

    std::vector<int> intersection; // to store the intersection of 2 vectors
    std::vector<int> my_union;     // to store the union of two vectors
    std::vector<int> diff;         // to store the difference between my_union & intersection

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        auto i = std::distance(observations.begin(), iter);

        std::sort(observations.at(i).begin(), observations.at(i).end());
        std::sort(predictions.at(i).begin(), predictions.at(i).end());

        // get intersection
        intersection.clear();
        std::set_intersection(observations[i].begin(), observations[i].end(), predictions[i].begin(), predictions[i].end(), std::back_inserter(intersection));
        // get union
        my_union.clear();
        std::set_union(observations[i].begin(), observations[i].end(), predictions[i].begin(), predictions[i].end(), std::back_inserter(my_union));
        // get difference
        diff.clear();
        std::set_difference(my_union.begin(), my_union.end(), intersection.begin(), intersection.end(), std::inserter(diff, diff.begin()));

        double num   = ((double) number_of_labels - (double) diff.size());
        double denom =  (double) number_of_labels;
        concordance =  num / denom;
        sum_of_concordances += concordance;

        int diff_size = diff.size();
        if (diff_size > number_of_labels) {
            PrintVector(observations[i]);
            PrintVector(predictions[i]);
            PrintVector(diff);
            std::cout << "number_of_labels = " << number_of_labels << ", diff.size() = " << diff.size() << std::endl;
            std::cout << "concordance = num " << num << "/ denom " << denom << " = " << concordance << " - sum = " << sum_of_concordances<< std::endl;
        }
    }

    double ratio = (double) sum_of_concordances / (double) sample_size;
    double hamming_loss = 1 - ratio;
    return hamming_loss;
}

/**
 * [Accuracy description]
 * @param observations 2D vector of label observations, each row a sample
 * @param predictions  2D vector of label predictions, each row a sample  [description]
 * @return [description]
 */
double Accuracy(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions)
{
    double ratio;
    double sum = 0;
    int count_null = 0; // count null intersections: very bad results
    std::vector<int> intersection;
    std::vector<int> my_union; // to store the union of two vectors

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        auto i = std::distance(observations.begin(), iter);

        std::sort(observations.at(i).begin(), observations.at(i).end());
        std::sort(predictions.at(i).begin(), predictions.at(i).end());

        // cardinality of intersection
        intersection.clear();
        std::set_intersection(observations[i].begin(), observations[i].end(),
                              predictions[i].begin(), predictions[i].end(),
                              std::back_inserter(intersection));

        if (intersection.size() == 0) {
          count_null++;
        }

        // cardinality of union
        my_union.clear();
        std::set_union(observations[i].begin(), observations[i].end(),
                       predictions[i].begin(), predictions[i].end(),
                       std::back_inserter(my_union));

        if (my_union.size() != 0) {
            ratio = (double) intersection.size() /
                    (double) my_union.size();
            sum += ratio;
        }
    }

    double result = (double) sum / (double) observations.size();
    return result;
}

/**
 * [Precision description]
 * @param observations 2D vector of label observations, each row a sample
 * @param predictions  2D vector of label predictions, each row a sample  [description]
 * @return [description]
 */
double Precision(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions)
{
    double ratio = 0;
    double sum = 0;
    std::vector<int> intersection;

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        // get "i", iteration index
        auto i = std::distance(observations.begin(), iter);
        // ratio = 0;
        if (predictions.at(i).size() > 0) {
            intersection.clear();
            std::sort(observations.at(i).begin(), observations.at(i).end());
            std::sort(predictions.at(i).begin(), predictions.at(i).end());

            // cardinality of intersection
            // vectors obs & pred must be SORTED
            std::set_intersection(observations[i].begin(), observations[i].end(),
            predictions[i].begin(), predictions[i].end(),
            std::back_inserter(intersection));

            ratio = (double) intersection.size() / (double) predictions[i].size();
            sum += ratio;
        }
    }

    double result = (double) sum / (double) observations.size();
    return result;
}

/**
 * [Recall description]
 * @param observations 2D vector of label observations, each row a sample
 * @param predictions  2D vector of label predictions, each row a sample
 * @return [description]
 */
double Recall(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions)
{
    double ratio = 0;
    double sum = 0;
    std::vector<int> intersection;

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        // get "i", iteration index
        auto i = std::distance(observations.begin(), iter);
        // ratio = 0;
        if (observations.at(i).size() > 0) {
            intersection.clear();
            std::sort(observations.at(i).begin(), observations.at(i).end());
            std::sort(predictions.at(i).begin(), predictions.at(i).end());

            // cardinality of intersection
            // vectors obs & pred must be SORTED
            std::set_intersection(observations[i].begin(), observations[i].end(),
            predictions[i].begin(), predictions[i].end(),
            std::back_inserter(intersection));

            ratio = (double) intersection.size() / (double) observations[i].size();
            sum += ratio;
        }
    }

    double result = (double) sum / (double) observations.size();
    return result;
}

/**
 * [F_measure description]
 * @param observations 2D vector of label observations, each row a sample
 * @param predictions  2D vector of label predictions, each row a sample
 * @param number_of_labels
 * @return [description]
 */
double F_measure(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels)
{
    double ratio;
    double sum = 0;
    int count_null = 0; // count null intersections: very bad results
    std::vector<int> intersection;
    int sum_of_cardinalities;

    for (auto iter = observations.begin(); iter != observations.end(); ++iter) {
        // get "i", iteration index
        auto i = std::distance(observations.begin(), iter);
        // obs & pred must be SORTED
        std::sort(observations.at(i).begin(), observations.at(i).end());
        std::sort(predictions.at(i).begin(), predictions.at(i).end());

        // cardinality of intersection
        intersection.clear();
        std::set_intersection(observations[i].begin(), observations[i].end(), predictions[i].begin(), predictions[i].end(),
            std::back_inserter(intersection));

        if (intersection.size() == 0) {
            count_null++;
        }
        sum_of_cardinalities = (observations[i].size() + predictions[i].size());
        if (sum_of_cardinalities != 0) {
            ratio = (double) intersection.size() / (double) sum_of_cardinalities;
            sum += ratio;
        }
    }

    double result = ((double) sum * (2 / (double) observations.size()));
    return result;
}
