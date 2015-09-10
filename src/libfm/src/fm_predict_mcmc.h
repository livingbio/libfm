#ifndef FM_PREDICT_MCMC_H_
#define FM_PREDICT_MCMC_H_
#include <sstream>


class fm_predict_mcmc : public fm_predict {
    public:
        virtual double evaluate(Data& data) { return std::numeric_limits<double>::quiet_NaN(); }
    protected:
        virtual double predict_case(Data& data) {
            throw "not supported for MCMC and ALS";
        }
    public:
        uint num_iter;
        uint num_eval_cases;

        double alpha_0, gamma_0, beta_0, mu_0;
        double alpha;

        double w0_mean_0;

        DVector<double> w_mu, w_lambda;

        DMatrix<double> v_mu, v_lambda;


        bool do_sample; // switch between choosing expected values and drawing from distribution
        bool do_multilevel; // use the two-level (hierarchical) model (TRUE) or the one-level (FALSE)
        uint nan_cntr_v, nan_cntr_w, nan_cntr_w0, nan_cntr_alpha, nan_cntr_w_mu, nan_cntr_w_lambda, nan_cntr_v_mu, nan_cntr_v_lambda;
        uint inf_cntr_v, inf_cntr_w, inf_cntr_w0, inf_cntr_alpha, inf_cntr_w_mu, inf_cntr_w_lambda, inf_cntr_v_mu, inf_cntr_v_lambda;

    protected:
        DVector<double> cache_for_group_values;
        sparse_row<DATA_FLOAT> empty_data_row; // this is a dummy row for attributes that do not exist in the training data (but in test data)

        DVector<double> pred_sum_all;
        DVector<double> pred_sum_all_but5;
        DVector<double> pred_this;

        e_q_term* cache_test;

        DVector<relation_cache*> rel_cache;

        virtual void _predict(Data& test) {};


        /**
            This function predicts all datasets mentioned in main_data.
            It stores the prediction in the e-term.
        */
        void predict_data_and_write_to_eterms(DVector<Data*>& main_data, DVector<e_q_term*>& main_cache) {
            assert(main_data.dim == main_cache.dim);
            if (main_data.dim == 0) { return ; }

            DVector<RelationJoin>& relation = main_data(0)->relation;

            // do this using only the transpose copy of the training data:
            for (uint ds = 0; ds < main_cache.dim; ds++) {
                e_q_term* m_cache = main_cache(ds);
                Data* m_data = main_data(ds);
                for (uint i = 0; i < m_data->num_cases; i++) {
                    m_cache[i].e = 0.0;
                    m_cache[i].q = 0.0;
                }
            }

            for (uint r = 0; r < relation.dim; r++) {
                for (uint c = 0; c < relation(r).data->num_cases; c++) {
                    rel_cache(r)[c].y = 0.0;
                    rel_cache(r)[c].q = 0.0;
                }
            }

            // (1) do the 1/2 sum_f (sum_i v_if x_i)^2 and store it in the e/y-term
            // (1.1) e_j = 1/2 sum_f (q_jf+ sum_R q^R_jf)^2
            // (1.2) y^R_j = 1/2 sum_f q^R_jf^2
            // Complexity: O(N_z(X^M) + \sum_{B} N_z(X^B) + n*|B| + \sum_B n^B) = O(\mathcal{C})
            for (int f = 0; f < fm->num_factor; f++) {
                double* v = fm->v.value[f];

                // calculate cache[i].q = sum_i v_if x_i (== q_f-term)
                // Complexity: O(N_z(X^M))
                for (uint ds = 0; ds < main_cache.dim; ds++) {
                    e_q_term* m_cache = main_cache(ds);
                    Data* m_data = main_data(ds);
                    m_data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
                        {
                            row_index = m_data->data_t->getRowIndex();
                            feature_data = &(m_data->data_t->getRow());
                            m_data->data_t->next();
                        }
                        double& v_if = v[row_index];

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            m_cache[train_case_index].q += v_if * x_li;
                        }
                    }
                }

                // calculate block_cache[i].q = sum_i v^B_if x^B_i (== q^B_f-term)
                // Complexity: O(\sum_{B} N_z(X^B))
                for (uint r = 0; r < relation.dim; r++) {
                    uint attr_offset = relation(r).data->attr_offset;
                    relation(r).data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < relation(r).data->data_t->getNumRows(); i++) {
                        {
                            row_index = relation(r).data->data_t->getRowIndex();
                            feature_data = &(relation(r).data->data_t->getRow());
                            relation(r).data->data_t->next();
                        }
                        double& v_if = v[row_index + attr_offset];

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            rel_cache(r)[train_case_index].q += v_if * x_li;
                        }
                    }

                }

                // add 0.5*q^2 to e and set q to zero.
                // O(n*|B|)
                for (uint ds = 0; ds < main_cache.dim; ds++) {
                    e_q_term* m_cache = main_cache(ds);
                    Data* m_data = main_data(ds);
                    for (uint c = 0; c < m_data->num_cases; c++) {
                        double q_all = m_cache[c].q;
                            for (uint r = 0; r < m_data->relation.dim; r++) {
                                q_all += rel_cache(r)[m_data->relation(r).data_row_to_relation_row(c)].q;
                            }
                        m_cache[c].e += 0.5 * q_all*q_all;
                        m_cache[c].q = 0.0;
                    }
                }


                // Calculate the "prediction" part of the relation y
                // O(\sum_B n^B)
                for (uint r = 0; r < relation.dim; r++) {
                    // add 0.5*q^2 to y and set q to zero.
                    for (uint c = 0; c <  relation(r).data->num_cases; c++) {
                        rel_cache(r)[c].y += 0.5 * rel_cache(r)[c].q * rel_cache(r)[c].q;
                        rel_cache(r)[c].q = 0.0;
                    }
                }
            }

            // (2) do -1/2 sum_f (sum_i v_if^2 x_i^2) and store it in the q-term
            for (int f = 0; f < fm->num_factor; f++) {
                double* v = fm->v.value[f];

                // sum up the q^S_f terms in the main-q-cache: 0.5*sum_i (v_if x_i)^2 (== q^S_f-term)
                // Complexity: O(N_z(X^M))
                for (uint ds = 0; ds < main_cache.dim; ds++) {
                    e_q_term* m_cache = main_cache(ds);
                    Data* m_data = main_data(ds);

                    m_data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
                        {
                            row_index = m_data->data_t->getRowIndex();
                            feature_data = &(m_data->data_t->getRow());
                            m_data->data_t->next();
                        }
                        double& v_if = v[row_index];

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            m_cache[train_case_index].q -= 0.5 * v_if * v_if * x_li * x_li;
                        }
                    }
                }

                // sum up the q^B,S_f terms in the block_cache.q: 0.5* sum_i (v^B_if x^B_i)^2 (== q^B,S_f-term)
                // Complexity: O(\sum_{B} N_z(X^B))
                for (uint r = 0; r < relation.dim; r++) {
                    uint attr_offset = relation(r).data->attr_offset;
                    relation(r).data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < relation(r).data->data_t->getNumRows(); i++) {
                        {
                            row_index = relation(r).data->data_t->getRowIndex();
                            feature_data = &(relation(r).data->data_t->getRow());
                            relation(r).data->data_t->next();
                        }
                        double& v_if = v[row_index + attr_offset];

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            rel_cache(r)[train_case_index].q -= 0.5 * v_if * v_if * x_li * x_li;
                        }
                    }
                }
            }

            // (3) add the w's to the q-term
            if (fm->k1) {
                for (uint ds = 0; ds < main_cache.dim; ds++) {
                    e_q_term* m_cache = main_cache(ds);
                    Data* m_data = main_data(ds);
                    m_data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < m_data->data_t->getNumRows(); i++) {
                        {
                            row_index = m_data->data_t->getRowIndex();
                            feature_data = &(m_data->data_t->getRow());
                            m_data->data_t->next();
                        }
                        double& w_i = fm->w(row_index);

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            m_cache[train_case_index].q += w_i * x_li;
                        }
                    }
                }
                for (uint r = 0; r < relation.dim; r++) {
                    uint attr_offset = relation(r).data->attr_offset;
                    relation(r).data->data_t->begin();
                    uint row_index;
                    sparse_row<DATA_FLOAT>* feature_data;
                    for (uint i = 0; i < relation(r).data->data_t->getNumRows(); i++) {
                        {
                            row_index = relation(r).data->data_t->getRowIndex();
                            feature_data = &(relation(r).data->data_t->getRow());
                            relation(r).data->data_t->next();
                        }
                        double& w_i = fm->w(row_index + attr_offset);

                        for (uint i_fd = 0; i_fd < feature_data->size; i_fd++) {
                            uint& train_case_index = feature_data->data[i_fd].id;
                            FM_FLOAT& x_li = feature_data->data[i_fd].value;
                            rel_cache(r)[train_case_index].q += w_i * x_li;
                        }
                    }
                }

            }
            // (3) merge both for getting the prediction: w0+e(c)+q(c)
            for (uint ds = 0; ds < main_cache.dim; ds++) {
                e_q_term* m_cache = main_cache(ds);
                Data* m_data = main_data(ds);

                for (uint c = 0; c < m_data->num_cases; c++) {
                    double q_all = m_cache[c].q;
                    for (uint r = 0; r < m_data->relation.dim; r++) {
                        q_all += rel_cache(r)[m_data->relation(r).data_row_to_relation_row(c)].q;
                    }
                    m_cache[c].e = m_cache[c].e + q_all;
                    if (fm->k0) {
                        m_cache[c].e += fm->w0;
                    }
                    m_cache[c].q = 0.0;
                }
            }

            // The "prediction" in each block is calculated
            for (uint r = 0; r < relation.dim; r++) {
                // y_i = y_i + q_i = [1/2 sum_f (q^B_if)^2] + [sum w^B_i x^B_i -1/2 sum_f (sum_i v^B_if^2 x^B_i^2)]
                for (uint c = 0; c <  relation(r).data->num_cases; c++) {
                    rel_cache(r)[c].y = rel_cache(r)[c].y + rel_cache(r)[c].q;
                    rel_cache(r)[c].q = 0.0;
                }
            }

        }
    public:




    public:
        virtual void outvalue(Data& data, DVector<double>& out) {
            assert(data.num_cases == out.dim);
            if (do_sample) {
                assert(data.num_cases == pred_sum_all.dim);
                for (uint i = 0; i < out.dim; i++) {
                    out(i) = pred_sum_all(i) / num_iter;
                }
            } else {
                assert(data.num_cases == pred_this.dim);
                for (uint i = 0; i < out.dim; i++) {
                    out(i) = pred_this(i);
                }
            }
            for (uint i = 0; i < out.dim; i++) {
                if (task == TASK_REGRESSION ) {
                    out(i) = std::min(max_target, out(i));
                    out(i) = std::max(min_target, out(i));
                } else if (task == TASK_CLASSIFICATION) {
                    out(i) = std::min(1.0, out(i));
                    out(i) = std::max(0.0, out(i));
                } else {
                    throw "task not supported";
                }
            }
        }


    public:
        virtual void init() {
            fm_predict::init();

            cache_for_group_values.setSize(meta->num_attr_groups);

            empty_data_row.size = 0;
            empty_data_row.data = NULL;

            alpha_0 = 1.0;
            gamma_0 = 1.0;
            beta_0 = 1.0;
            mu_0 = 0.0;

            alpha = 1;

            w0_mean_0 = 0.0;

            w_mu.setSize(meta->num_attr_groups);
            w_lambda.setSize(meta->num_attr_groups);
            w_mu.init(0.0);
            w_lambda.init(0.0);

            v_mu.setSize(meta->num_attr_groups, fm->num_factor);
            v_lambda.setSize(meta->num_attr_groups, fm->num_factor);
            v_mu.init(0.0);
            v_lambda.init(0.0);


            if (log != NULL) {
                log->addField("alpha", std::numeric_limits<double>::quiet_NaN());
                if (task == TASK_REGRESSION) {
                    log->addField("rmse_mcmc_this", std::numeric_limits<double>::quiet_NaN());
                    log->addField("rmse_mcmc_all", std::numeric_limits<double>::quiet_NaN());
                    log->addField("rmse_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());

                    //log->addField("rmse_mcmc_test2_this", std::numeric_limits<double>::quiet_NaN());
                    //log->addField("rmse_mcmc_test2_all", std::numeric_limits<double>::quiet_NaN());
                } else if (task == TASK_CLASSIFICATION) {
                    log->addField("acc_mcmc_this", std::numeric_limits<double>::quiet_NaN());
                    log->addField("acc_mcmc_all", std::numeric_limits<double>::quiet_NaN());
                    log->addField("acc_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());
                    log->addField("ll_mcmc_this", std::numeric_limits<double>::quiet_NaN());
                    log->addField("ll_mcmc_all", std::numeric_limits<double>::quiet_NaN());
                    log->addField("ll_mcmc_all_but5", std::numeric_limits<double>::quiet_NaN());

                    //log->addField("acc_mcmc_test2_this", std::numeric_limits<double>::quiet_NaN());
                    //log->addField("acc_mcmc_test2_all", std::numeric_limits<double>::quiet_NaN());
                }

                std::ostringstream ss;
                for (uint g = 0; g < meta->num_attr_groups; g++) {
                    ss.str(""); ss << "wmu[" << g << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
                    ss.str(""); ss << "wlambda[" << g << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
                    for (int f = 0; f < fm->num_factor; f++) {
                        ss.str(""); ss << "vmu[" << g << "," << f << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
                        ss.str(""); ss << "vlambda[" << g << "," << f << "]"; log->addField(ss.str(), std::numeric_limits<double>::quiet_NaN());
                    }
                }
            }
        }


        virtual void predict(Data& test) {
            pred_sum_all.setSize(test.num_cases);
            pred_sum_all_but5.setSize(test.num_cases);
            pred_this.setSize(test.num_cases);
            pred_sum_all.init(0.0);
            pred_sum_all_but5.init(0.0);
            pred_this.init(0.0);

            // init caches data structure
            MemoryLog::getInstance().logNew("e_q_term", sizeof(e_q_term), test.num_cases);
            cache_test = new e_q_term[test.num_cases];

            _predict(test);

            MemoryLog::getInstance().logFree("e_q_term", sizeof(e_q_term), test.num_cases);
            delete[] cache_test;
        }


        virtual void debug() {
            fm_predict::debug();
            std::cout << "do_multilevel=" << do_multilevel << std::endl;
            std::cout << "do_sampling=" << do_sample << std::endl;
            std::cout << "num_eval_cases=" << num_eval_cases << std::endl;
        }

};

#endif /*FM_PREDICT_MCMC_H_*/
