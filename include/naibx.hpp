/**@file: naibx.hpp
 *
 * @author Luca Mossina
 */

#ifndef NAIBX_HPP_
#define NAIBX_HPP_

#ifndef PI
#define PI 3.14159265359
#endif

#include <vector>
#include <iostream>
#include <ostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm> // find(), sort()
#include <numeric>
#include <cstdio>
#include <cmath>
#include <string>
#include "managedata.hpp"   // PrintVector()
#include <map>
#include <cassert>
#include "candidate.hpp"

// void try_inserting(std::multimap<double, Candidate>, Candidate candy);
// bool does_not_exist_in_keep_cand(Candidate new_cand, std::multimap<double, Candidate> keep_cand);
// bool is_candidate_full(Candidate candy);
// bool exists_a_not_full_candidate(std::multimap<double, Candidate> keep_cand);
/* print Candidates */
// void PrintVector(std::vector<Candidate> const &vec) {
//     std::vector<int> labels;
//     for (size_t i = 0; i < vec.size(); i++) {
//         labels.clear();
//         labels = vec.at(i).get_label_set();
//         std::cout << "m = " << vec.at(i).size << " --- labels = "; PrintVector(labels);
//     }
// }


struct Naibx {
    ///////////////////////////////////////////////////////////////////////////////
    // List here all the parameters taken by the learning & prediction algorithm //
    ///////////////////////////////////////////////////////////////////////////////
    int N_set; // SAMPLE SIZE.
    std::vector<int> N_m; // 1 x (|Y|+ 1) - array storing the num of examples for  M = m.
    int n_features; // dimension of feature space, e.g. R x R = R² --> n_features = 2.
    int n_labels; // dimension of target set. corresponds to "nb_class"
    double log_epsilon; // epsilon, used in smoothing
    bool laplacian_smoothing;
    std::vector< std::vector<double> > mu_im; // n_features x (|Y|+ 1)
    std::vector< std::vector<double> > M2_im; // n_features x (|Y|+ 1)
    std::vector<int> count_labels; // 1 x |Y| "array storing the num of examples seen for y € Y"
    std::vector< std::vector<double> > mu_iy; // n_features x (|Y|)
    std::vector< std::vector<double> > M2_iy; // n_features x (|Y|)
    std::vector< std::vector<int> > count_yy; // |Y| x |Y|, num of (y'€Y|y€Y). count_yy[i][i] = 0
    std::vector< std::vector<int> > N_ym; // |Y| x (|Y|+ 1). count occurences of (M = m | y € Y)
    /* NEW ATTRIBUTES */
    // IGNORE THIS
    int keep; // pruning param: number of best scores to keep

    // Constructor //

    /// laplacian_ = true: Laplacian smoothing
    Naibx(int n_features_ = 1, int n_labels_ = 2, bool laplacian_ = false,  int keep_ = 3, double epsilon_ = 1e-16)
    :
    N_set(0), // SAMPLE SIZE
    N_m((n_labels_ + 1), 0), // vector of zeroes,
    n_features(n_features_), // dimension of feature space
    n_labels(n_labels_), //|Y|, dim of target space
    log_epsilon(log(epsilon_)),
    laplacian_smoothing(laplacian_),
    mu_im(n_features_, std::vector<double>((n_labels_ + 1), 0.)), // stores the mean of P(X_i = x_i| M = m)
    M2_im(n_features_, std::vector<double>((n_labels_ + 1), 0.)),
    count_labels(n_labels_, 0), // aka N_count_y. // WARNING!! For small sample size, if count_labels[i] <= 1, then prob = -nan
    mu_iy(n_features_, std::vector<double>(n_labels_, 0.)),
    M2_iy(n_features_, std::vector<double>(n_labels_, 0.)),
    count_yy(n_labels_, std::vector<int>(n_labels_, 0)),
    N_ym(n_labels_, std::vector<int>((n_labels_ + 1), 0)),
    keep(keep_)
    {}

    /**
     * add_example()  Algorithm 4:  learning step for Naive Bayes ensemble
     * prediction. Each iterations update the parameters of the predictor.
     * This member function is called in a loop in Main(), for each sample in learning set.
     * @param x values from <X> = (X_1, X_2, ...X_i, ... X_n). X_i's are continuous.
     * @param y  Labels <y_obs> = (y_1, y_2, ... y_k) associated to <X>_obs.
     */
    void add_example(const std::vector<double> &x_pred,
                     const std::vector<int> &y_target)
    {
        //(1) Learn SIZES. update MEANS & VARIANCES
        int m = y_target.size();

        // SAMPLE SIZE. Init = 0
        N_set++;

        // presumably different "m" are going to be observed. Here we keep track of each target "y" for which a length "m" was observed. The most reasonable container is a simple vector, i.e. a vector of frequencies.
        // ?? OBS: dimension ov vector is |Y|+1, because m = 0 is an admissible target parameter??

        // EXAMPLE: if length(y) = m = 2: N_m[2]=[0, 0, +1, 0, ...]
        N_m[m]++;
        // update MEAN & VARIANCE
        for (int i = 0; i < n_features; ++i) {
            double delta = x_pred[i] - mu_im[i][m];
            mu_im[i][m] += (delta / ((double) N_m[m]));
            M2_im[i][m] += (delta * (x_pred[i] - mu_im[i][m]));
        }

        // (2) Learn the elements of x_pred.
        for (int j = 0; j < m; ++j) {
            // TARGET VALUE
            int y = y_target[j];

            count_labels[y] = count_labels[y] + 1;
            N_ym[y][m]++;

            for (int i = 0; i < n_features; ++i) {
                double delta = x_pred[i] - mu_iy[i][y];
                mu_iy[i][y] += (delta/count_labels[y]);
                M2_iy[i][y] += (delta * (x_pred[i] - mu_iy[i][y]));
            }

            for (int k = 0; k < m; ++k) {
                if (k != j) {
                    int y_p = y_target[k];
                    ++(count_yy[y][y_p]);
                }
                else {continue;}
            }
        }
    }

    /**
     * predict_m() This function computes the a score for each possible size of the target vector and returns the "keep" best ones.
     * @param observed_x   observed values for explicative variables X
     * @param target_sizes vector of the best (ordered) sizes for the target vector to be predicted
     * @param scores       for each possible size m of the target vector, it stores its correspondent score (i.e. the numerator in bayes formulation)
     * @return A pointer to itself
     */
    Naibx& predict_m(
        const std::vector<double> & observed_x,
              std::vector<int>    & target_sizes,
              std::vector<double> & scores,
                    const bool    & admits_zero = false)
    {
        // we start with the worst possible score: -inf
        double min_val = -std::numeric_limits<double>::infinity();

        // keep: attribute of naibx, num of best sizes to be stored
		std::vector<int>(keep).swap(target_sizes);
        std::vector<double> (n_labels + 1, min_val).swap(scores);

        double var = 0.; // variance
        double log_density_XM;

        double logpx = 0; // store DENOMINATOR: log(P(X))

        std::vector<double> numerator(n_labels + 1, 0); // numerator of bayes formula, one value for each possible size, including empty size

        double denom = 0; // denominator of bayes formula

        /* fill in all scores */
        int m;
        if (admits_zero) {
            m = 0;
        } else {
            m = 1;
        }
        for (; m <= n_labels; ++m) {
        // for (int m = 0; m <= n_labels; ++m) {

            /* SCORE 1 - P(M) */
            double score1 = 0;
            if(laplacian_smoothing) {
				score1 = log((double) (N_m[m] + 1)) - log((double) (N_set + n_labels + 1));
                // ?? shouldn't be:
				// score1 = log((double) (N_m[m] + 1)) - log((double) (N_set + n_labels));
			}
			else {
				score1 = log((double) N_m[m]) - log((double) N_set);
			}

            /* SCORE 2 - P(X|M)*/
            double score2 = 0;
            if (N_m[m] > 1) {
				for (int i = 0; i < n_features; i++) {
					var = M2_im[i][m] / (double)(N_m[m] - 1);
					log_density_XM = log_normal_pdf(mu_im[i][m], var , observed_x[i]);
                    score2 += log_density_XM;
				}
                logpx += score2;
			}
			else {
                for (int i = 0; i < n_features; i++) {
                    var = 100000; // TODO: must take care of this
                    log_density_XM = log_normal_pdf(0, var , observed_x[i]);
                    score2 += log_density_XM;
                }
                logpx += score2;
			}
            scores[m] = score1 + score2;

            /* The numerator of bayes formulation: P(M=m | X=x) = num[m]/P(X=x) */
            numerator[m] = exp(scores[m]);
        }

        /* denom = P(X=x) = (NB assumption) = Sum(numerator[i]) */
        denom = 0;
        for (size_t i = 0; i < numerator.size(); i++) {
            denom += numerator.at(i);
        }

        // compute pseudo-prob (naive bayes assumption)
        std::vector<double> probix(numerator.size(), 0); // contains probs for each
        for (size_t i = 0; i < numerator.size(); i++) {
            probix.at(i) = numerator.at(i) / denom;
        }

        // sum all probabilities
        double tot_prob = 0;
        for (std::vector<double>::iterator it = probix.begin(); it != probix.end(); ++it)
            tot_prob += *it;

        /* STANDARD WAY of computing best m */
        /**
         *  find the best K scores. Return their corresponding "m"
         *  1) find MAX score
         *  2) push back corresponding position
         *  3) set it to min (-inf)
         *  4) loop "keep" times
         */
        // std::vector<double>::iterator result; // store partial results in loop
        // int position; // "i" index of elem whose score is max
        //
        // for (int i = 0; i < keep; i++) {
        //     // result: points at max element
        //     result = std::max_element(scores.begin(), scores.end());
        //
        //     // position: get position of max in vector
        //     // "position" corresponds to an "m" value, inserted into "target_sizes", the vector of best "m"
        //     position = std::distance(scores.begin(), result);
        //
        //     target_sizes.at(i) = position;
        //     // set current max to min, to be IGNORED at next iteration
        //     *result = min_val;
        // }

        /* ALTERNATIVE WAY: PSEUDO PROBABILITIES */
        /**
         *  find the best K probabilities. Return their corresponding "m"
         *  1) find MAX prob
         *  2) push back corresponding position
         *  3) set it to min (zero)
         *  4) loop "keep" times
         */
        std::vector<double>::iterator result; // store partial results in loop
        int position; // "i" index of elem whose score is max

        for (int i = 0; i < keep; i++) {
            result = std::max_element(probix.begin(), probix.end()); // points at max element

            // position: get position of max in vector
            // "position" corresponds to an "m" value, inserted into "target_sizes", the vector of best "m"
            position = std::distance(probix.begin(), result);

            target_sizes.at(i) = position;
            // set current max to min, to be IGNORED at next iteration
            *result = min_val;
        }
        return *this;
    }


    /* NaiBX */
    int predict_yk(const std::vector<double> &observed_x,
                   const int &size,
                   const std::vector<int> &predicted_so_far,
                   const std::vector<int> &available_labels )
    {
        // int k = 1 + predicted_so_far.size();
        int k = predicted_so_far.size();

        double best_score = -std::numeric_limits<double>::infinity();
        int best_label = 0;
        double var = 0.;

        double log_py, log_nm, log_p_my;
        double log_p_jy;

        std::vector<double> scores(n_labels, 0.);
        double log_prob_iy;

        for (int y : available_labels) {

            /* P(Y) */
            log_py = log((double) count_labels.at(y)) - log((double) N_set);

            scores.at(y) = log_py;

            /* P(X_i|Y) */
            for (int i = 0; i < n_features; i++) {
                var = M2_iy[i].at(y) / (double)(count_labels.at(y) - 1);

                log_nm = (log_normal_pdf(mu_iy[i].at(y), var, observed_x[i]));
                // log_prob_iy[i].at(y) = log_nm;
                log_prob_iy = log_nm;
                scores.at(y) = scores.at(y) + log_prob_iy;
            }

            /* P(M|Y) */
            log_p_my = log((double) N_ym.at(y)[size]) - log((double) count_labels.at(y));
            // std::cout << "log_p_" << size << "_" << y << " = " << log_p_my << std::endl;
            scores.at(y) = scores.at(y) + log_p_my;

            /* P(Y'|Y) */
            // WARNING HERE
            for (int j = 0; j < k; j++) {
                int y0 = predicted_so_far.at(j);
                log_p_jy = log((double) count_yy.at(y).at(y0)) - log((double) count_labels.at(y));
                scores.at(y) = scores.at(y) + log_p_jy;
            }

            // WHAT IF there are elements with equivalently best score?
            if (scores.at(y) > best_score) {
                best_score = scores.at(y);
                best_label = y;
            }
        }
        return best_label;
    }

    std::vector<int> predict_y(
        const std::vector<double> &observed_x,
              std::vector<int>    &y_pred,
        const int & obs_m,
        const bool & real_m)
    {
        int target_size;
        int temp_label;

        y_pred.clear();

        if (real_m == true) {
            target_size = obs_m;
        }
        else {
            std::vector<int> m_size(keep);
            std::vector<double> scores(n_labels + 1);

            // predict the "keep" best m sizes
            predict_m(observed_x, m_size, scores);
            target_size = m_size.at(0);
        }

        std::vector<int> available_labels(n_labels);
        // std::iota => Fills the range [first, last) with sequentially increasing values
        std::iota (std::begin(available_labels), std::end(available_labels), 0);

        for (int k = 1; k <= target_size; k++) {
            temp_label = predict_yk(observed_x, target_size, y_pred, available_labels);
            std::vector<int>::iterator I = std::find(available_labels.begin(), available_labels.end(), temp_label);
            if (I != available_labels.end()) {
                available_labels.erase(I);
            }

            y_pred.push_back(temp_label);
        }
        std::sort(y_pred.begin(), y_pred.end());

        return y_pred;
    }


    /* NaiBX - top m */
    /* Returns 2d array with K solutions, one for each of the top k "m" */
    std::vector<std::vector<int> > predict_y(
        const std::vector<double> & observed_x,
              std::vector<int>    & y_pred)
    {
        std::vector<std::vector<int> > keep_cand(keep);

        int target_size; // vector of target sizes
        int temp_label;
        std::vector<int> m_size(keep);
        std::vector<double> scores(n_labels + 1);
        predict_m(observed_x, m_size, scores);

        // std::vector<int> available_labels(n_labels, 0);
        // // fill the range [first, last) with sequentially increasing values
        // std::iota (std::begin(available_labels), std::end(available_labels), 0);

        // loop on the best "keep" sizes
        for (int current_size = 0; current_size < keep; ++current_size) {
            y_pred.clear();
            scores.clear();

            // fill the range [first, last) with sequentially increasing values
            std::vector<int> available_labels(n_labels, 0);
            std::iota (std::begin(available_labels), std::end(available_labels), 0);

            target_size = m_size.at(current_size);
            // if (current_size == (keep-1)) {
            //     std::cout << "current_size = " << target_size << std::endl;
            // }

            for (int k = 0; k < target_size; k++) {
                temp_label = predict_yk(observed_x, target_size, y_pred, available_labels);

                std::vector<int>::iterator I = std::find(available_labels.begin(), available_labels.end(), temp_label);
                if (I != available_labels.end()) {
                    available_labels.erase(I);
                }

                // if (current_size == (keep-1)) {
                //     std::cout << "temp_label = " << temp_label << std::endl;
                // }
                y_pred.push_back(temp_label);
            }

            keep_cand.at(current_size) = y_pred;
        }

        return keep_cand;
    }


    /* KBEST VERSION */
    Naibx& predict_m(
        const std::vector<double> &observed_x,
              std::vector<double> &scores)
    {
        double min_val = -std::numeric_limits<double>::infinity();

        std::vector<double> (n_labels + 1, min_val).swap(scores);

        double var = 0.; // variance
        double log_density_XM;
        double logpx = 0; // store DENOMINATOR: log(P(X))

        for (int m = 0; m <= n_labels; ++m) {
            double score1 = 0;
            if(laplacian_smoothing) {
				score1 = log((double) (N_m[m] + 1)) - log((double) (N_set + n_labels + 1));
                // ?? shouldn't be:
				// score1 = log((double) (N_m[m] + 1)) - log((double) (N_set + n_labels));
			}
			else {
				score1 = log((double) N_m[m]) - log((double) N_set);
			}
            double score2 = 0;
            if (N_m[m] > 1) {
				for (int i = 0; i < n_features; i++) {
					var = M2_im[i][m] / (double)(N_m[m] - 1);
					log_density_XM = log_normal_pdf(mu_im[i][m], var , observed_x[i]);
                    score2 += log_density_XM;
				}
                logpx += score2;
			}
			else {
                for (int i = 0; i < n_features; i++) {
                    var = 100000;
                    log_density_XM = log_normal_pdf(0, var , observed_x[i]);
                    score2 += log_density_XM;
                }
                logpx += score2;
			}
            scores[m] = score1 + score2;
        }
        return *this;
    }

    /* KBEST VERSION */
    // IGNORE THIS: OLD EXPERIMENT
    std::vector<Candidate> predict_yk(
        const std::vector<double> & observed_x,
        const Candidate cand)
    {
        std::vector<Candidate> list_cand;
        std::vector<int> predicted_so_far;

        predicted_so_far = cand.get_label_set();
        int k = predicted_so_far.size();

        std::vector<int> available_labels(n_labels);
        std::iota (std::begin(available_labels), std::end(available_labels), 0);

        // get rid of used labels
        for (size_t i = 0; i < predicted_so_far.size(); i++) {
            available_labels.erase(
                std::remove(available_labels.begin(), available_labels.end(), predicted_so_far.at(i) ), available_labels.end());
        }

        double var = 0.;
        double log_py, /*log_nm,*/ log_p_my, log_p_jy, log_prob_iy;
        double score;
        // std::vector<double> scores(n_labels + 1, 0.);
        std::vector<double> scores(n_labels, 0.);

        Candidate temp_cand;

        for (int y : available_labels) {
            temp_cand = cand;

            /* P(Y) */
            // log_py = log((double) count_labels.at(y)) - log((double) N_set);
                // laplacian_smoothing
            log_py = log((double) count_labels.at(y) + 1) - log((double) N_set + (n_labels + 1));
            score = log_py;

            /* P(X_i|Y) */
            if (count_labels.at(y) > 1) {
                for (int i = 0; i < n_features; i++) {
                    var = M2_iy[i].at(y) / (double)(count_labels.at(y) - 1);
                    log_prob_iy = (log_normal_pdf(mu_iy[i].at(y), var, observed_x[i]));
                    score = score + log_prob_iy;
                }
            }
            else {
                for (int i = 0; i < n_features; i++) {
                    var = 100000;
                    log_prob_iy = (log_normal_pdf(mu_iy[i].at(y), var, observed_x[i]));
                    score = score + log_prob_iy;
                }
            }

            /* P(M|Y) */
            // log_p_my = log((double) N_ym.at(y)[cand.size]) - log((double) count_labels.at(y));
                // laplacian_smoothing
            log_p_my = log((double) N_ym.at(y)[cand.size] + 1) - log((double) count_labels.at(y) + (n_labels + 1));
            score = score + log_p_my;

            /* P(Y'|Y) */
            for (int j = 0; j < k; j++) {
                int y0 = predicted_so_far.at(j);
                log_p_jy = log((double) count_yy.at(y).at(y0)) - log((double) count_labels.at(y));
                score = score + log_p_jy;
            }

            scores.at(y) = score;

            temp_cand.score = (scores.at(y));
            // std::cout << "temp_cand.score = " << temp_cand.score << std::endl;
            temp_cand.add_label(y);

            list_cand.push_back(temp_cand);
        }
        // std::cout << "scores = "; PrintVector(scores);
        return list_cand;
    }

    // std::multimap<double, Candidate> kbest(const std::vector<double> &observed_x)
    std::vector<std::vector<int> > predict_y(const std::vector<double> &observed_x)
    {
        // std::cout << " ------------------ " << std::endl;
        std::vector<double> scores;
        std::multimap<double, Candidate> keep_cand;
        std::vector<std::vector<int> > array_cand;

        predict_m(observed_x, scores);

        /* print best k "m" */
        std::vector<double>::iterator result; // store partial results in loop
        // int position; // "i" index of elem whose score is max
        for (int i = 0; i < keep; i++) {
            // result: points at max element
            result = std::max_element(scores.begin(), scores.end());

            // position: get position of max in vector
            // "position" corresponds to an "m" value, inserted into "target_sizes", the vector of best "m"
            // position = std::distance(scores.begin(), result);

            // target_sizes.at(i) = position;
            // set current max to min, to be IGNORED at next iteration
            *result = -100000000;
        }

        for (int size = 0; size <= n_labels; size++) {
            Candidate candy(size); // init candidate of m = "size"
            candy.score = (scores.at(size));
            try_inserting(keep_cand, candy);
        }

        std::multimap<double, Candidate> old_keep_cand;
        std::multimap<double, Candidate>::iterator it;
        std::vector<Candidate> list_of_candidates;

        while (exists_a_not_full_candidate(keep_cand)) {
            old_keep_cand = keep_cand;
            keep_cand.clear();

            for (it = old_keep_cand.begin(); it != old_keep_cand.end(); ++it) {
                Candidate candy;
                candy = (*it).second;
                if (is_candidate_full(candy)) {
                    try_inserting(keep_cand, candy);
                }
                else { // if NOT full
                    list_of_candidates = predict_yk(observed_x, candy);

                    for (Candidate coco : list_of_candidates) {
                        if (does_not_exist_in_keep_cand(coco, keep_cand)) {
                            try_inserting(keep_cand, coco);
                        }
                    }
                }
            }
        }

        for (it = keep_cand.begin(); it != keep_cand.end(); ++it) {
            array_cand.push_back((*it).second.get_label_set());
        }
        return array_cand;
    }

    // IGNORE THIS: OLD EXPERIMENT
    void try_inserting(std::multimap<double, Candidate> & keep_cand,
                                   Candidate  new_cand)
    {
        std::multimap<double, Candidate>::iterator it;
        int keep_cand_size = keep_cand.size();
        if (keep_cand_size < keep) {
            keep_cand.insert(
                std::pair<double, Candidate>(new_cand.score, new_cand));
        }
        else {
            // OBS: in multimap, lowest index is on top, i.e; multimap.begin()
            it = keep_cand.begin();
            if (new_cand.score > (*it).first) {
                /* pop first one out*/
                keep_cand.erase(it);

                /* insert new_cand at position new_cand.score in keep_cand */
                keep_cand.insert(
                    std::pair<double, Candidate>(new_cand.score, new_cand));
            }
            assert(keep_cand_size == keep && "keep_cand (multimap) contains more than k candidates");
        }
    }

    // IGNORE THIS: OLD EXPERIMENT
    bool does_not_exist_in_keep_cand(Candidate new_cand,
        std::multimap<double, Candidate> keep_cand)
    {
        std::multimap<double, Candidate>::iterator old_cand;

        for (old_cand = keep_cand.begin(); old_cand!=keep_cand.end(); ++old_cand) {
            if (new_cand.is_equal((*old_cand).second)) {
                return false;
            }
        }
        return true;
    }

    // IGNORE THIS: OLD EXPERIMENT
    bool is_candidate_full(Candidate candy)
    {
        int candy_label_set_size = candy.label_set.size();
        return (candy_label_set_size == candy.size);
    }

    // IGNORE THIS: OLD EXPERIMENT
    bool exists_a_not_full_candidate(std::multimap<double,Candidate> keep_cand)
    {
        std::multimap<double, Candidate>::iterator it;

        for (it = keep_cand.begin(); it!=keep_cand.end(); ++it) {
            if (!is_candidate_full((*it).second)) {
                return true;
            }
        }
        return false;
    }

    /* helper functions */
    double log_normal_pdf(double mu, double var, double x)
    {return -(pow(mu - x, 2))/(2 * var) - .5 * log(2 * PI * var);}

    /**
     * save_model() create text file with training parameters values]
     * @param filename title of text file with results of training
     */
    void save_model(std::string filename)
    {
		std::ofstream file(filename);

        file << "N_set\n" << N_set << std::endl;
        file << "n_features\n" << n_features << std::endl;
        file << "n_labels\n" << n_labels << std::endl;
        file << "log_epsilon\n" << log_epsilon << std::endl;

        file << "N_m " << std::endl;
        for (size_t i = 0; i < N_m.size(); i++) {
            file << N_m.at(i) << " ";
        }

        file << "\ncount_labels " << std::endl;
        for (size_t i = 0; i < count_labels.size(); i++) {
            file << count_labels[i] << " ";
        }

        file << "\nmu_im" << std::endl;
        for (size_t i = 0; i < mu_im.size(); i++) {
            for (size_t j = 0; j < mu_im[i].size(); j++) {
                file << mu_im[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

        file << "M2_im" << std::endl;
        for (size_t i = 0; i < M2_im.size(); i++) {
            for (size_t j = 0; j < M2_im[i].size(); j++) {
                file << M2_im[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

        // file << "sum_labels_obs\n" << sum_labels_obs << std::endl;

        file << "mu_iy" << std::endl;
        for (size_t i = 0; i < mu_iy.size(); i++) {
            for (size_t j = 0; j < mu_iy[i].size(); j++) {
                file << mu_iy[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

        file << "M2_iy" << std::endl;
        for (size_t i = 0; i < M2_iy.size(); i++) {
            for (size_t j = 0; j < M2_iy[i].size(); j++) {
                file << M2_iy[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

        file << "count_yy "<< std::endl;
        for (size_t i = 0; i < count_yy.size(); i++) {
            for (size_t j = 0; j < count_yy[i].size(); j++) {
                file << count_yy[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

        file << "N_ym " << std::endl;
        for (size_t i = 0; i < N_ym.size(); i++) {
            for (size_t j = 0; j < N_ym[i].size(); j++) {
                file << N_ym[i].at(j) << " ";
            }
            file << "" << std::endl;
        }

		file << std::endl;
		file.close();
	}

    void load_my_model(const std::string &filename)
    {
        clear_values(); // not necessary if you are not mixing stuff

        std::ifstream input_data(filename); //, std::ios::in);

        // if model could not be loaded, interrupt
        if (!input_data.is_open()) { // test whether opening succeeded
            std::cerr << "Input model could not be loaded (for prediction)" << std::endl;
            // std::cerr << "Error: " << strerror(errno) << std::endl;
            return;
        }

        std::string line; // store line-by-line in data file
        std::stringstream liness; // string stream line
        std::string word; // container for elements in line

        // start reading model

        /* N_set */
        std::getline(input_data, line); // reads new line
        liness.clear(); // clears previous contents of liness
        liness.str(line); // store line into liness

        liness >> word; // store first word, it should be "N_set"
        if (word != "N_set") {
            std::cerr << "expecting N_set, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear();
        liness.str(line);
        liness >> N_set;

        /* n_features */
        std::getline(input_data, line);
        liness.clear();liness.str(line);

        liness >> word;
        if (word != "n_features") {
            std::cerr << "expecting n_features, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> n_features;

        /* n_labels */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "n_labels") {
            std::cerr << "expecting n_labels, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> n_labels;

        /* log_epsilon */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "log_epsilon") {
            std::cerr << "expecting log_epsilon, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> log_epsilon;

        /* N_m */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "N_m") {
            std::cerr << "expecting N_m, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear();
        liness.str(line); //
        for (int i = 0; i < (n_labels+1); i++) {
            int val;
            liness >> val;
            N_m.at(i) = val;
        }

        /* count_labels */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "count_labels") {
            std::cerr << "expecting count_labels, received: " << word << std::endl;
            return;
        }
        std::getline(input_data, line);
        liness.clear();
        liness.str(line);
        for (int i = 0; i < (n_labels); i++) {
            int val;
            liness >> val;
            count_labels.at(i) = val;
        }

        /* mu_im */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "mu_im") {
            std::cerr << "expecting mu_im, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_features; i++) {
            std::getline(input_data, line);
            liness.clear(); liness.str(line);
            for (int j = 0; j < (n_labels + 1); j++) {
                double val;
                liness >> val;
                mu_im.at(i).at(j) = val;
            }
        }

        /* M2_im */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);

        liness >> word;
        if (word != "M2_im") {
            std::cerr << "expecting M2_im, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_features; i++) {
            std::getline(input_data, line);
            liness.clear(); liness.str(line);
            for (int j = 0; j < (n_labels + 1); j++) {
                double val;
                liness >> val;
                M2_im.at(i).at(j) = val;
            }
        }

        // //////////////
        // // sum_labels_obs //
        // ////////////
        // std::getline(input_data, line);
        // liness.clear(); liness.str(line);
        //
        // liness >> word;
        // if (word != "sum_labels_obs") {
        //     std::cerr << "expecting sum_labels_obs, received: " << word << std::endl;
        //     return;
        // }
        // std::getline(input_data, line);
        // liness.clear(); liness.str(line);
        // liness >> sum_labels_obs;

        /* mu_iy */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> word;
        if (word != "mu_iy") {
            std::cerr << "expecting mu_iy, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_features; i++) {
            std::getline(input_data, line);
            liness.clear();liness.str(line);
            for (int j = 0; j < (n_labels); j++) {
                double val;
                liness >> val;
                mu_iy.at(i).at(j) = val;
            }
        }

        /* M2_iy */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> word;
        if (word != "M2_iy") {
            std::cerr << "expecting M2_iy, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_features; i++) {
            std::getline(input_data, line);
            liness.clear(); liness.str(line);
            for (int j = 0; j < (n_labels); j++) {
                double val;
                liness >> val;
                M2_iy.at(i).at(j) = val;
            }
        }

        /* count_yy */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> word;
        if (word != "count_yy") {
            std::cerr << "expecting count_yy, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_labels; i++) {
            std::getline(input_data, line);
            liness.clear(); liness.str(line);
            for (int j = 0; j < (n_labels); j++) {
                int val;
                liness >> val;
                count_yy.at(i).at(j) = val;
            }
        }

        /* N_ym */
        std::getline(input_data, line);
        liness.clear(); liness.str(line);
        liness >> word;
        if (word != "N_ym") {
            std::cerr << "expecting N_ym, received: " << word << std::endl;
            return;
        }
        for (int i = 0; i < n_labels; i++) {
            std::getline(input_data, line);
            liness.clear(); liness.str(line);
            for (int j = 0; j < (n_labels + 1); j++) {
                int val;
                liness >> val;
                N_ym.at(i).at(j) = val;
            }
        }
    }

    // clear values
    void clear_values()
    {
        n_features = 0;
        n_labels = 0;
        // log_epsilon = 0.;
        // log_epsilon=log(epsilon);
        // std::vector<double>().swap(mu_iy);
        // std::vector<double>().swap(M2);
        // std::vector<int>().swap(count_xd);
        // std::vector<int>().swap(nb_points_Y);
        // nb_points=0;
    }




};
#endif
