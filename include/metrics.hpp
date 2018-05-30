#ifndef METRICS_HPP_
#define METRICS_HPP_

#include <vector>
#include <iostream>
#include <string>

void ExportMetrics(
    std::string my_arff,
    std::string metr,
    double loss_value,
    bool realm);

double LCard(
    std::vector<std::vector<int> > target_lab);

double F_measure(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels);

double Recall(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions);

double Precision(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions);

double Accuracy(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions);

double HammingLoss(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels);

double ZeroOneLoss(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions);

void LabelAccuracy(
    std::vector<std::vector<int> > observations,
    std::vector<std::vector<int> > predictions,
    int number_of_labels);


#endif
