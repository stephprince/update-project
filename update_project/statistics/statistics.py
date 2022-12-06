import itertools
import numpy as np
import pandas as pd
import pingouin as pg
import warnings

from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path
from scipy.stats import sem, ranksums, kstest
from tqdm import tqdm

# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri

from update_project.general.results_io import ResultsIO
#
# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages(ro.StrVector(['lme4', 'lmerTest', 'emmeans', 'report']))
# lme4 = importr('lme4')
# lme4= importr('lmerTest')  # TODO - determine if I need this or not
# emmeans = importr('emmeans')
# report = importr('report')

rng = np.random.default_rng(12345)

class Stats:
    def __init__(self, levels=None, approaches=None, tests=None, alternatives=None, results_io=None):
        # setup defaults unless given otherwise
        self.levels = levels or ['animal', 'session_id']  # append levels needed (e.g., trials, units)
        self.approaches = approaches or ['bootstrap', 'traditional']  # 'summary' as other option
        self.tests = tests or ['direct_prob', 'mann-whitney']  # 'wilcoxon' as other option
        self.alternatives = alternatives or ['two-sided']  # 'greater', 'less' as other options
        self.nboot = 1000  # number of iterations to perform for bootstrapping default should be 1000
        self.results_io = results_io or ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def run(self, df, dependent_vars=None, group_vars='group', pairs=None, filename=''):
        # function requires data to be a dataframe in the following format
        # levels, dependent variable column, group variable column for group comparison
        self.dependent_vars = dependent_vars
        self.group_vars = group_vars
        self.pairs = pairs

        stats_output = []
        descript_output = []
        for a, t, alt in itertools.product(self.approaches, self.tests, self.alternatives):
            if a == 'bootstrap' and t != 'direct_prob':  # only run bootstrapping for direct prob test
                continue
            else:
                self._setup_data(approach=a, data=df)

                descript = self._get_descriptive_stats(approach=a, test=t, alternative=alt)
                descript_output.append(descript)

                stats = self._perform_test(approach=a, test=t, alternative=alt, pairs=pairs)
                stats_output.append(stats)

        self.stats_df = pd.concat(stats_output, axis=0)
        self.descript_df = pd.concat(descript_output, axis=0)

        self._export_stats(filename)

    def _setup_data(self, approach, data):
        if approach == 'bootstrap':
            self.df_processed = (data
                                 .groupby(self.group_vars)
                                 .apply(lambda grp: self._get_bootstrapped_data(grp))
                                 .reset_index())
        elif approach == 'summarized':
            self.df_processed = data.groupby(
                [*self.group_vars, self.levels[0]]).mean().reset_index()  # calc means for level 1
        elif approach == 'traditional':
            self.df_processed = data.copy(deep=True)
        elif approach == 'mixed_effects':
            self.df_processed = self._get_mixed_effects_model(data)
        else:
            warnings.warn(f'Statistical approach {approach} is not supported')
            self.df_processed = []

        return self

    def _perform_test(self, approach, test, alternative, pairs=None):
        pair_outputs = []
        for p in pairs:
            query = [' & '.join([f'{g_item} == "{p_item}"' if isinstance(p_item, str)
                                 else f'{g_item} == {p_item}' for p_item, g_item in zip(p_var, self.group_vars)])
                     for p_var in p]
            samples = [self.df_processed.query(q) for q in query]

            for var in self.dependent_vars:
                if test == 'direct_prob':
                    comparisons = {0: f'{p[1]} >= {p[0]}', 1: f'{p[0]} >= {p[1]}'}
                    prob_vals = (self.get_direct_prob(samples[0][var].to_numpy(), samples[1][var].to_numpy()),
                                 self.get_direct_prob(samples[1][var].to_numpy(), samples[0][var].to_numpy()))
                    test_output = dict(pair=p, variable=var, test=test, approach=approach, prob_test_vals=[prob_vals],
                                       p_val=np.min(prob_vals) * 2, alternative=comparisons[np.argmin(prob_vals)])
                elif test == 'mann-whitney':
                    output = pg.mwu(samples[0][var].to_numpy(), samples[1][var].to_numpy(), alternative=alternative)
                    test_output = dict(pair=p, variable=var, test=test, approach=approach,
                                       prob_test_vals=output['U-val'].to_numpy()[0],
                                       p_val=output["p-val"].to_numpy()[0],
                                       alternative=output['alternative'].to_numpy()[0])
                elif test == 'wilcoxon':
                    output = pg.wilcoxon(samples[0][var].to_numpy(), samples[1][var].to_numpy(), alternative=alternative)
                    test_output = dict(pair=p, variable=var, test=test, approach=approach,
                                       prob_test_vals=output['W-val'].to_numpy()[0],
                                       p_val=output["p-val"].to_numpy()[0],
                                       alternative=output['alternative'].to_numpy()[0])
                elif test == 'anova':
                    output = pg.mixed_anova()  # TODO - determine if I need mixed or not

                pair_outputs.append(test_output)

        return pd.DataFrame(pair_outputs)

    def _get_descriptive_stats(self, approach, test, alternative):
        descriptive_stats = (self.df_processed
                              .groupby(self.group_vars)[self.dependent_vars]
                              .describe()
                              .reset_index()
                              .melt(id_vars=[(g, '') for g in self.group_vars], value_name='metric',
                                    var_name=['variable_0', 'variable_1'])
                              .pipe(lambda x: x.set_axis(x.columns.map(''.join), axis=1))
                              .pivot(columns='variable_1', index=[*self.group_vars, 'variable_0'],
                                     values='metric')
                              .reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1)
                              .reset_index())
        descriptive_stats.insert(0, 'approach', approach)
        descriptive_stats.insert(1, 'test', test)
        descriptive_stats.insert(2, 'alternative', alternative)

        return descriptive_stats

    def _export_stats(self, filename):
        self.results_io.export_statistics(self.descript_df, f'{filename}_descriptive', format='csv')
        self.results_io.export_statistics(self.stats_df, f'{filename}_p_values', format='csv')

    def _get_mixed_effects_model(self, data):
        data['predictor'] = data[self.group_vars].apply(tuple, axis=1)
        for var in self.dependent_vars:

            formula = f'{var} ~ predictor + (1|animal) + (1|animal:session_id)'  # equivalent to var ~ predictor + 1|animal/session_id
            model = lme4.lmer(formula, data=self.pandas_to_r_df(data), REML=False)
            print(ro.r['summary'](model))
            print(ro.r['anova'](model))
            print(emmeans.emmeans(model, 'predictor', contr='pairwise', adjust='tukey'))
            # model2 = ro.r['lm'](f'{var} ~ predictor', data=self.pandas_to_r_df(data))
            # print(ro.r['anova'](model, model2))  # just for curiosity to compare with/without random effects
            #print(report.report(model))

            # TODO - add checks for assumptions of mixed effects models
            # TODO - add plots for visualization of data
            # TODO - add stats output in dataframe format
            # not super clear but I'm pretty sure it's the same formula as 'var ~ predictor + (1|animal/session_id)' like above
            model3 = (MixedLM
                      .from_formula(f'{var} ~ C(predictor)', vc_formula={'session_id': '0+C(session_id)'}, re_formula='1',
                                    groups='animal', data=data)
                      .fit(method=['lbfgs'], reml=False))
            print(model3.summary())
            re_df = pd.DataFrame.from_dict(model3.random_effects, orient='index')
            re_df['intercept'] = re_df['animal'] + model3.fe_params.loc['Intercept']
            re_df['slopes'] = re_df.iloc[:, 1:] + model3.fe_params.loc['Intercept']

        return output_data

    @staticmethod
    def pandas_to_r_df(data):
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(data)

        return r_df

    def _bootstrap_recursive(self, data, current_level=0, output_data=None):
        unique_samples = np.unique(data[self.levels[current_level]])
        random_samples = rng.choice(unique_samples, len(unique_samples))

        if current_level == 0:
            output_data = []

        for samp in random_samples:
            samp = f'"{samp}"' if isinstance(samp, str) else samp
            subset = data.query(f'{self.levels[current_level]} == {samp}')

            if current_level == len(self.levels) - 1:  # if it's the last level, add data
                output_data.append(subset[self.dependent_vars])
            else:  # if it's not the last level, keep going down hierarchy and resampling
                output_data = self._bootstrap_recursive(subset, current_level=current_level + 1,
                                                        output_data=output_data)

        return output_data

    def _get_bootstrapped_data(self, data, fxn=None):
        '''
        This function performs a hierarchical bootstrap
        This function assumes that the data is a dataframe where levels indicates column containing
        data from highest hierarchical level to lowest. Data is assumed to already be separated by group
        so only one group is input to the function at a time and the first level is resampled with replacement
        '''

        bootstats = []
        for i in tqdm(range(self.nboot), desc='bootstrap'):
            bootstrapped_data = pd.concat(self._bootstrap_recursive(data), axis=0)
            if fxn:
                calculation_output = fxn(bootstrapped_data)  # default to calculate mean but can apply any function
            else:
                calculation_output = bootstrapped_data.mean()

            bootstats.append(calculation_output)

        return pd.concat(bootstats, axis=1).transpose()

    @staticmethod
    def get_direct_prob(sample1, sample2):
        '''
        get_direct_prob Returns the direct probability of items from sample2 being
        greater than or equal to those from sample1.
           Sample1 and Sample2 are two bootstrapped samples and this function
           directly computes the probability of items from sample 2 being greater
           than those from sample1. Since the bootstrapped samples are
           themselves posterior distributions, this is a way of computing a
           Bayesian probability. The joint matrix can also be returned to compute
           directly upon.
        '''
        joint_low_val = min([min(sample1), min(sample2)])
        joint_high_val = max([max(sample1), max(sample2)])

        p_joint_matrix = np.zeros((100, 100))
        p_axis = np.linspace(joint_low_val, joint_high_val, num=100)
        edge_shift = (p_axis[2] - p_axis[1]) / 2
        p_axis_edges = p_axis - edge_shift
        p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

        # Calculate probabilities using histcounts for edges.

        p_sample1 = np.histogram(sample1, bins=p_axis_edges)[0] / np.size(sample1)
        p_sample2 = np.histogram(sample2, bins=p_axis_edges)[0] / np.size(sample2)

        # Now, calculate the joint probability matrix:

        for i in np.arange(np.shape(p_joint_matrix)[0]):
            for j in np.arange(np.shape(p_joint_matrix)[1]):
                p_joint_matrix[i, j] = p_sample1[i] * p_sample2[j]

        # Normalize the joint probability matrix:
        p_joint_matrix = p_joint_matrix / np.sum(p_joint_matrix)

        # Get the volume of the joint probability matrix in the upper triangle:
        p_test = np.sum(np.triu(p_joint_matrix, k=1))  # k=1 calculate greater than instead of greater than or equal to

        return p_test


def get_fig_stats(data, axis=0):
    if len(data):  # if not empty
        mean = np.nanmean(data, axis=axis)
        err = sem(data, axis=axis, nan_policy='omit')
        upper = mean + 2 * err
        lower = mean - 2 * err
        err_upper = mean + err
        err_lower = mean - err
    else:
        mean = []
        err = []
        upper = []
        lower = []
        err_upper = []
        err_lower = []
        # nan_arr = np.empty(np.delete(np.shape(data), axis))

    stats_dict = dict(mean=mean,
                      err=err,
                      upper=upper,
                      lower=lower,
                      err_upper=err_upper,
                      err_lower=err_lower
                      )

    return stats_dict


def get_descriptive_stats(data, name=None, axis=0):
    mean = np.nanmean(data, axis=axis)
    median = np.nanmedian(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    err = sem(data, axis=axis)
    upper = mean + 2 * err  # 95% CI
    lower = mean - 2 * err  # 95% CI
    quantiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1])
    n = np.shape(data)[axis]

    stats_dict = dict(mean=mean,
                      median=median,
                      std=std,
                      err=err,
                      upper=upper,
                      lower=lower,
                      quantiles=quantiles,
                      n=n,
                      )

    return stats_dict


def get_comparative_stats(x, y):
    rs_test_statistic, rs_p_value = ranksums(x, y)
    ks_test_statistic, ks_p_value = kstest(x, y)

    stats_dict = dict(ranksum=dict(test_statistic=rs_test_statistic,
                                   p_value=rs_p_value, ),
                      kstest=dict(test_statistic=ks_test_statistic,
                                  p_value=ks_p_value, ),
                      )

    return stats_dict


def get_stats_summary(data_dict, axis=0):
    plus_minus = '\u00B1'
    new_line = '\n'

    summary_list = []
    keys = []
    for key, value in data_dict.items():
        stats = get_descriptive_stats(value, axis)
        keys.append(key)
        summary_list.append(f"{key}: {stats['mean']} {plus_minus} {stats['err']}, n = {stats['n']}, "
                            f"quantiles = {stats['quantiles']}{new_line}")

    comparative_stats = get_comparative_stats(*data_dict.values())
    for key, value in comparative_stats.items():
        summary_list.append(f"{key}: p = {value['p_value']}, test-statistic = {value['test_statistic']}{new_line}")

    summary_text = ''.join(summary_list)

    return summary_text
