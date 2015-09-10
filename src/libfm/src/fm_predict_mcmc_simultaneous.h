#ifndef FM_PREDICT_MCMC_SIMULTANEOUS_H_
#define FM_PREDICT_MCMC_SIMULTANEOUS_H_

#include "fm_predict_mcmc.h"


class fm_predict_mcmc_simultaneous : public fm_predict_mcmc {
    protected:

        virtual void _predict(Data& test) {
            DVector<Data*> main_data(1);
            DVector<e_q_term*> main_cache(1);
            main_data(0) = &test;
            main_cache(0) = cache_test;

            // predict test and train
            predict_data_and_write_to_eterms(main_data, main_cache);


            if (task == TASK_REGRESSION) {
                // evaluate test and store it
                for (uint c = 0; c < test.num_cases; c++) {
                    double p = cache_test[c].e;
                    pred_this(c) = p;
                    p = std::min(max_target, p);
                    p = std::max(min_target, p);
                    pred_sum_all(c) += p;
                    pred_sum_all_but5(c) += p;
                }
            } else if (task == TASK_CLASSIFICATION) {
                // evaluate test and store it
                for (uint c = 0; c < test.num_cases; c++) {
                    double p = cache_test[c].e;
                    p = cdf_gaussian(p);
                    pred_this(c) = p;
                    pred_sum_all(c) += p;
                    pred_sum_all_but5(c) += p;
                }
            }
        }

        void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint from_case, uint to_case) {
            assert(pred.dim == target.dim);
            double _rmse = 0;
            double _mae = 0;
            uint num_cases = 0;
            for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
                double p = pred(c) * normalizer;
                p = std::min(max_target, p);
                p = std::max(min_target, p);
                double err = p - target(c);
                _rmse += err*err;
                _mae += std::abs((double)err);
                num_cases++;
            }

            rmse = std::sqrt(_rmse/num_cases);
            mae = _mae/num_cases;

        }

        void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint from_case, uint to_case) {
            double _loglikelihood = 0.0;
            uint _accuracy = 0;
            uint num_cases = 0;
            for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
                double p = pred(c) * normalizer;
                if (((p >= 0.5) && (target(c) > 0.0)) || ((p < 0.5) && (target(c) < 0.0))) {
                    _accuracy++;
                }
                double m = (target(c)+1.0)*0.5;
                double pll = p;
                if (pll > 0.99) { pll = 0.99; }
                if (pll < 0.01) { pll = 0.01; }
                _loglikelihood -= m*log10(pll) + (1-m)*log10(1-pll);
                num_cases++;
            }
            loglikelihood = _loglikelihood/num_cases;
            accuracy = (double) _accuracy / num_cases;
        }


        void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint& num_eval_cases) {
            _evaluate(pred, target, normalizer, rmse, mae, 0, num_eval_cases);
        }

        void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint& num_eval_cases) {
            _evaluate_class(pred, target, normalizer, accuracy, loglikelihood, 0, num_eval_cases);
        }
};

#endif /*FM_PREDICT_MCMC_SIMULTANEOUS_H_*/
