/**@file: bow_nbx.hpp
 *
 * @author Luca Mossina
 * Created: 17 Nov 2016
 *
 * An extension of NaiBX to handle Bag-of-Words input data
 */

#ifndef BOWNAIBX_HPP_
#define BOWNAIBX_HPP_

#include <vector>
#include <iostream>
#include <ostream>
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

struct BowNaibx {
    int N_set; // SAMPLE SIZE.
    std::vector<int> N_m; // 1 x (|Y|+ 1) - array storing the num of examples for  M = m.
    int n_features; // dimension of feature space, e.g. R x R = R² --> n_features = 2.
    int n_labels; // dimension of target set. corresponds to "nb_class"
    double log_epsilon; // epsilon, used in smoothing
    bool laplacian_smoothing;
    std::vector<std::vector<double> > pr_im; // n_features x (|Y|+ 1)
    std::vector<std::vector<int> > word_im; // n_features x (|Y|+ 1)
    std::vector<int> count_labels; // 1 x |Y| "array storing the num of examples seen for y € Y"
    std::vector<std::vector<double> > pr_iy; // n_features x (|Y|)
    std::vector<std::vector<int> > word_iy; // n_features x (|Y|)
    std::vector< std::vector<int> > count_yy; // |Y| x |Y|, num of (y'€Y|y€Y). count_yy[i][i] = 0
    std::vector< std::vector<int> > N_ym; // |Y| x (|Y|+ 1). count occurences of (M = m | y € Y)
    int keep; // pruning param: number of best scores to keep

    BowNaibx(int n_features_ = 1, int n_labels_ = 2, bool laplacian_ = false,  int keep_ = 3, double epsilon_ = 1e-16)
    :
    N_set(0), // SAMPLE SIZE
    N_m((n_labels_ + 1), 0), // vector of zeroes, m=0 is admissible
    n_features(n_features_), // dimension of feature space
    n_labels(n_labels_), //|Y|, dim of target space
    log_epsilon(log(epsilon_)),
    laplacian_smoothing(laplacian_),
    pr_im(n_features_, std::vector<double>((n_labels_ + 1), 1.)),
    word_im(n_features_, std::vector<int>((n_labels_ + 1), 0)),
    count_labels(n_labels_, 0), // aka N_count_y. // WARNING!! For small sample size, if count_labels[i] <= 1, then prob = -nan
    pr_iy(n_features_, std::vector<double>(n_labels_, 1.)),
    word_iy(n_features_, std::vector<int>(n_labels_, 0)),
    count_yy(n_labels_, std::vector<int>(n_labels_, 0)),
    N_ym(n_labels_, std::vector<int>((n_labels_ + 1), 0)),
    keep(keep_)
    {}

    void add_example(
        const std::map<int, int> &my_dict_W,
        const std::vector<int>   &y_target)
    {
        /* (1) Learn SIZES. update probs & frequencies*/
        const int m = y_target.size();

        N_set++;
        N_m.at(m)++;

        for (int i = 0; i < n_features; ++i) {
            // if the word was observed (is non-zero)
            if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                word_im.at(i).at(m)++;
            }
        }

        /* (2) Learn the elements of x_pred. */
        for (int j = 0; j < m; ++j) {
            // TARGET VALUE
            int y = y_target[j];

            count_labels.at(y) = count_labels.at(y) + 1;
            N_ym.at(y).at(m)++;

            for (int i = 0; i < n_features; ++i) {
                if (my_dict_W.find( i ) != my_dict_W.end()) {
                    word_iy.at(i).at(y) += my_dict_W.at(i);
                }
            }

            for (int k = 0; k < m; ++k) {
                if (k != j) {
                    int y_p = y_target[k];
                    ++(count_yy.at(y).at(y_p));
                }
                else {continue;}
            }
        }
    }

    BowNaibx& predict_m(const std::map<int, int> &my_dict_W,
                           std::vector<int> &target_sizes,
                           std::vector<double> &scores)
    {
        // we start with the worst possible score: -inf
        double min_val = -std::numeric_limits<double>::infinity();

        // keep: attribute of BowNaibx, num of best sizes to be stored
		std::vector<int>(keep).swap(target_sizes);
        std::vector<double> (n_labels + 1, min_val).swap(scores);

        double score1 = 0;
        double score2 = 0;
        double proba  = 0.;
        // int count_present = 0;
        int exponent;

        /* fill in all scores */
        for (int m = 0; m <= n_labels; ++m) {
            // count_present = 0; // count bow non-zero entries
            /* SCORE 1 - P(M) */
            score1 = 0;

            /* Estimate priors from data */
            if(laplacian_smoothing) {
				score1 = log((double) (N_m.at(m) + 1)) - log((double) (N_set + n_labels + 1));
			}
			else {
				score1 = log((double) N_m.at(m)) - log((double) N_set);
			}

            /* SCORE 2 - P(W|M)*/
            score2 = 0;

            /* P(W_i=w_i|M=m) */
            for (int i = 0; i < n_features; i++) {
                if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                    // here always non-zero (lapl sm)
                    proba =   log((double) word_im.at(i).at(m) + 1 )
                            - log((double) N_m.at(m) + (n_features + 1));
                    exponent = my_dict_W.at(i);
                    score2 += exponent * proba;
                }
                // score2 += proba;
            }

            scores.at(m) = score1 + score2;
        }

        /* STANDARD WAY of computing best m */
        /**
         *  find the best K scores. Return their corresponding "m"
         *  1) find MAX score
         *  2) push back corresponding position
         *  3) set it to min (-inf)
         *  4) loop "keep" times
         */
        std::vector<double>::iterator result; // store partial results in loop
        int position; // "i" index of elem whose score is max

        for (int i = 0; i < keep; i++) {
            // result: points at max element
            result = std::max_element(scores.begin(), scores.end());
            // "position" corresponds to an "m" value, inserted into "target_sizes", the vector of best "m"
            position = std::distance(scores.begin(), result);
            target_sizes.at(i) = position;
            // set current max to min, to be IGNORED at next iteration
            *result = min_val;
        }
        return *this;
    }

    /* BowNaibx */
    int predict_yk(const std::map<int, int> &my_dict_W,
                   const int &size,
                   const std::vector<int> &predicted_so_far,
                   const std::vector<int> &available_labels )
    {
        // int k = 1 + predicted_so_far.size();
        int k = predicted_so_far.size();

        double best_score = -std::numeric_limits<double>::infinity();
        int best_label = 0;

        double log_py, /*log_nm,*/ log_p_my;
        double log_p_jy;

        // int count_present = 0;

        // std::vector<double> scores(n_labels, 0.);
        std::vector<double> scores(n_labels, -std::numeric_limits<double>::infinity());
        // double log_prob_iy;
        double proba = 0;

        for (int y : available_labels) {
            scores.at(y) = 0;
            /* P(Y): Prior Prob */
            log_py =   log((double) count_labels.at(y) + 1)
                     - log((double) N_set              + n_labels);
            scores.at(y) += log_py;

            // count_present = 0;

            /* P(W_i|Y)*/
            for (int i = 0; i < n_features; i++) {
                // IF word was observed in new document,
                // pick its estimated conditional probability
                if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                    // here always non-zero (lapl sm)
                    proba =   log((double) word_iy.at(i).at(y) + 1 )
                            - log((double)count_labels.at(y)   + n_features);
                    int exponent = my_dict_W.at(i);
                    scores.at(y) += exponent * proba;
                }
                // ELSE exponent=0, hence score=0, hence no need to add to scores.at()
            }

            /* P(M|Y) */
            log_p_my =   log((double) (N_ym.at(y).at(size) + 1) )
                       - log((double) (count_labels.at(y)  + n_labels));
            scores.at(y) += + log_p_my;

            /* P(Y'|Y) */
            // WARNING HERE
            for (int j = 0; j < k; j++) {
                int y0 = predicted_so_far.at(j);
                log_p_jy =   log((double) count_yy.at(y).at(y0) + 1)
                           - log((double) count_labels.at(y)    + n_labels);
                scores.at(y) += log_p_jy;
            }

            // WHAT IF there are elements with equivalently best score?
            // TODO: if (score_yi == score_yj): pick y with HIGHEST PRIOR
            if (scores.at(y) > best_score) {
                best_score = scores.at(y);
                best_label = y;
            }
        }
        return best_label;
    }

    std::vector<int> predict_y(const std::map<int, int> &my_dict_W,
                                     std::vector<int> &y_pred,
                               const int &obs_m,
                               const bool &real_m)
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

            predict_m(my_dict_W, m_size, scores);
            target_size = m_size.at(0);
        }

        std::vector<int> available_labels(n_labels);
        std::iota (std::begin(available_labels), std::end(available_labels), 0);

        for (int k = 1; k <= target_size; k++) {
            temp_label = predict_yk(my_dict_W, target_size, y_pred, available_labels);
            std::vector<int>::iterator I = std::find(available_labels.begin(), available_labels.end(), temp_label);
            if (I != available_labels.end()) {
                available_labels.erase(I);
            }
            y_pred.push_back(temp_label);
        }
        std::sort(y_pred.begin(), y_pred.end());

        return y_pred;
    }

    void save_model(std::string filename)  {
        std::cout << "WARNING: Bownaibx::save_model() NOT IMPLEMENTED" << std::endl;
        return;
    }
    void load_my_model(const std::string &filename) {
        std::cout << "WARNING: Bownaibx::load_my_model() NOT IMPLEMENTED" << std::endl;
        return;
    }
    void clear_values() {return;}
};
struct BernoulliBowNaibx {
    int N_set; // SAMPLE SIZE.
    std::vector<int> N_m; // 1 x (|Y|+ 1) - array storing the num of examples for  M = m.
    int n_features; // dimension of feature space, e.g. R x R = R² --> n_features = 2.
    int n_labels; // dimension of target set. corresponds to "nb_class"
    double log_epsilon; // epsilon, used in smoothing
    bool laplacian_smoothing;
    std::vector<std::vector<double> > pr_im; // n_features x (|Y|+ 1)
    std::vector<std::vector<int> > countdocs_im; // n_features x (|Y|+ 1)
    std::vector<int> count_labels; // 1 x |Y| "array storing the num of examples seen for y € Y"
    std::vector<std::vector<double> > pr_iy; // n_features x (|Y|)
    std::vector<std::vector<int> > countdocs_iy; // n_features x (|Y|)
    std::vector< std::vector<int> > count_yy; // |Y| x |Y|, num of (y'€Y|y€Y). count_yy[i][i] = 0
    std::vector< std::vector<int> > N_ym; // |Y| x (|Y|+ 1). count occurences of (M = m | y € Y)
    int keep; // pruning param: number of best scores to keep

    BernoulliBowNaibx(int n_features_ = 1,
                      int n_labels_ = 2,
                      bool laplacian_ = false,
                      int keep_ = 3,
                      double epsilon_ = 1e-16)
    :
    N_set(0), // SAMPLE SIZE
    N_m((n_labels_ + 1), 0), // vector of zeroes, m=0 is admissible
    n_features(n_features_), // dimension of feature space
    n_labels(n_labels_), //|Y|, dim of target space
    log_epsilon(log(epsilon_)),
    laplacian_smoothing(laplacian_),
    pr_im(n_features_, std::vector<double>((n_labels_ + 1), 1.)),
    countdocs_im(n_features_, std::vector<int>((n_labels_ + 1), 0)),
    count_labels(n_labels_, 0), // aka N_count_y. // WARNING!! For small sample size, if count_labels[i] <= 1, then prob = -nan
    pr_iy(n_features_, std::vector<double>(n_labels_, 1.)),
    countdocs_iy(n_features_, std::vector<int>(n_labels_, 0)),
    count_yy(n_labels_, std::vector<int>(n_labels_, 0)),
    N_ym(n_labels_, std::vector<int>((n_labels_ + 1), 0)),
    keep(keep_)
    {}

    void add_example(
        const std::map<int, int> &my_dict_W,
        const std::vector<int>   &y_target)
    {
        /* (1) Learn SIZES. update probs & frequencies*/
        const size_t m = y_target.size();

        N_set++;
        N_m.at(m)++;

        for (int i = 0; i < n_features; ++i) {
            // if the word was observed (is non-zero)
            if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                countdocs_im.at(i).at(m)++;
            }
        }

        /* (2) Learn the elements of x_pred. */
        for (size_t j = 0; j < y_target.size(); ++j) {
            // TARGET VALUE
            int this_class = y_target[j];

            count_labels.at(this_class) += 1;
            N_ym.at(this_class).at(m)++;

            for (int token = 0; token < n_features; ++token) {
                if (my_dict_W.find(token) != my_dict_W.end()) {
                    countdocs_iy.at(token).at(this_class) += my_dict_W.at(token);
                }
            }

            for (size_t k = 0; k < m; ++k) {
                if (k != j) {
                    int y_p = y_target[k];
                    ++(count_yy.at(this_class).at(y_p));
                }
                else {continue;}
            }
        }
    }

    BernoulliBowNaibx& predict_m(const std::map<int, int> &my_dict_W,
                           std::vector<int>    &target_sizes,
                           std::vector<double> &scores)
    {
        // we start with the worst possible score: -inf
        double min_val = -std::numeric_limits<double>::infinity();

        // keep: attribute of BowNaibx, num of best sizes to be stored
		std::vector<int>(keep).swap(target_sizes);
        std::vector<double> (n_labels + 1, min_val).swap(scores);

        double score = 0;
        double proba  = 0.;
        // int count_present = 0;
        int exponent = 1;

        /* fill in all scores */
        for (int m = 0; m <= n_labels; ++m) {
            // count_present = 0; // count bow non-zero entries
            /* SCORE 1 - P(M) */
            score = 0;

            /* Estimate priors from data */
            if(laplacian_smoothing) {
                score = log((double) (N_m.at(m) + 1)) - log((double) (N_set + n_labels + 1));
			}
			else {
				score = log((double) N_m.at(m)) - log((double) N_set);
			}

            /* SCORE 2 - P(W|M)*/
            /* P(W_i=w_i|M=m) */
            for (int i = 0; i < n_features; i++) {
                if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                    // here always non-zero (lapl sm)
                    proba =   log((double) countdocs_im.at(i).at(m) + 1 )
                            - log((double) N_m.at(m) + (n_features + 1));
                    exponent = my_dict_W.at(i);
                    // scores.at(m) += exponent * proba;
                    score += (((double) exponent) * proba);
                }
                score += proba;
            }
            scores.at(m) = score;
        }

        std::vector<double>::iterator result; // store partial results in loop
        int position; // "i" index of elem whose score is max

        for (int i = 0; i < keep; i++) {
            // result: points at max element
            result = std::max_element(scores.begin(), scores.end());
            // "position" corresponds to an "m" value, inserted into "target_sizes", the vector of best "m"
            position = std::distance(scores.begin(), result);
            target_sizes.at(i) = position;
            // set current max to min, to be IGNORED at next iteration
            *result = min_val;
        }
        return *this;
    }

    /* Bernoulli BowNaibx */
    int predict_yk(const std::map<int, int> &my_dict_W,
                   const int &size,
                   const std::vector<int> &predicted_so_far,
                   const std::vector<int> &available_labels )
    {
        // int k = 1 + predicted_so_far.size();
        int k = predicted_so_far.size();

        double best_score = -std::numeric_limits<double>::infinity();
        int best_label = 0;

        double log_py, /*log_nm,*/ log_p_my;
        double log_p_jy;

        // int count_present = 0;

        // std::vector<double> scores(n_labels, 0.);
        std::vector<double> scores(n_labels, -std::numeric_limits<double>::infinity());
        // double log_prob_iy;
        double proba = 0.;
        double score = 0.;

        double my_a = 0.;
        double my_b = 0.;

        for (int y : available_labels) {
            score = 0.;
            // scores.at(y) = 0;
            /* P(Y): Prior Prob */
            log_py =   log((double) count_labels.at(y) + 1)
                     - log((double) N_set              + n_labels);
            score += log_py;

            // count_present = 0;

            /* P(W_i|Y)*/

            // // FASTER (BUGGY) VERSION
            // int this_token = 0;
            // for (auto && kv : my_dict_W) {
            //     if (this_token < n_features) {
            //         // IF token is  NOT in map
            //         for ( ; this_token < kv.first; ++this_token) {
            //             // log(1 - a/b) = log( (b - a)/b ) = log(b - a) - log(b)
            //             // Here, a & b are:
            //             my_a = ((double) countdocs_iy.at(this_token).at(y) + 1); // <-- bernoulli stuff
            //             my_b = ((double) count_labels.at(y) + 2);
            //             proba = log( my_b - my_a) - log(my_b);
            //             score +=  proba;
            //         }
            //
            //         // ELSE IF token was observed:
            //         proba =   log((double) countdocs_iy.at(this_token).at(y) + 1 )
            //         - log((double) count_labels.at(y)                + 2 ); // <-- bernoulli stuff
            //         score +=  proba;
            //
            //         ++this_token;
            //     }
            // }

            for (int i = 0; i < n_features; i++) {
                // IF word was observed in new document,
                // pick its estimated conditional probability
                if ( my_dict_W.find( i ) != my_dict_W.end() ) {
                    proba =   log((double) countdocs_iy.at(i).at(y) + 1 )
                            - log((double) count_labels.at(y)       + 2 ); // <-- bernoulli stuff
                    score +=  proba;
                }
                else {
                    // log(1 - a/b) = log( (b - a)/b ) = log(b - a) - log(b)
                    my_a = ((double) countdocs_iy.at(i).at(y) + 1); // <-- bernoulli stuff
                    my_b = ((double) count_labels.at(y) + 2);
                    proba = log( my_b - my_a) - log(my_b);
                    score +=  proba;
                }
            }

            /* P(M|Y) */
            log_p_my =   log((double) (N_ym.at(y).at(size) + 1) )
                       - log((double) (count_labels.at(y)  + n_labels));
            score += + log_p_my;

            /* P(Y'|Y) */
            for (int j = 0; j < k; j++) {
                int y0 = predicted_so_far.at(j);
                log_p_jy =   log((double) count_yy.at(y).at(y0) + 1)
                           - log((double) count_labels.at(y)    + n_labels);
                score += log_p_jy;
            }

            scores.at(y) = score;

            // WHAT IF there are elements with equivalently best score?
            // TODO: if (score_yi == score_yj): pick y with HIGHEST PRIOR
            if (scores.at(y) > best_score) {
                best_score = scores.at(y);
                best_label = y;
            }
        }
        return best_label;
    }

    std::vector<int> predict_y(const std::map<int, int> &my_dict_W,
                                     std::vector<int> &y_pred,
                               const int obs_m,
                               const bool real_m)
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
            predict_m(my_dict_W, m_size, scores);
            target_size = m_size.at(0);
        }

        std::vector<int> available_labels(n_labels);
        std::iota (std::begin(available_labels), std::end(available_labels), 0);

        for (int k = 1; k <= target_size; k++) {
            temp_label = predict_yk(my_dict_W, target_size, y_pred, available_labels);
            std::vector<int>::iterator I = std::find(available_labels.begin(), available_labels.end(), temp_label);
            if (I != available_labels.end()) {
                available_labels.erase(I);
            }
            y_pred.push_back(temp_label);
        }
        std::sort(y_pred.begin(), y_pred.end());

        return y_pred;
    }

    void save_model(std::string filename)  { return;}
    void load_my_model(const std::string &filename) { return;}
    void clear_values() {return;}
};
#endif
