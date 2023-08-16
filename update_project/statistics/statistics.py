import itertools
import numpy as np
import pandas as pd
import pingouin as pg
import warnings

from ast import literal_eval
from pathlib import Path
from scipy.stats import sem, ranksums, kstest, spearmanr
from tqdm import tqdm

import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import pandas2ri

from update_project.general.results_io import ResultsIO

utils = importr('utils')
utils.chooseCRANmirror(ind=1)
package_names = ['lme4', 'lmerTest', 'emmeans', 'report', 'pbkrtest']
packages_to_install = [x for x in package_names if not isinstalled(x)]
utils.install_packages(ro.StrVector(packages_to_install))
lme4 = importr('lme4')
lme4 = importr('lmerTest')
emmeans = importr('emmeans')
report = importr('report')

rng = np.random.default_rng(12345)
new_line = '\n'


class Stats:
    def __init__(self, levels=None, approaches=None, tests=None, alternatives=None, results_io=None, units=None,
                 nboot=1000, results_type=None):
        # setup defaults unless given otherwise
        self.levels = levels or ['animal', 'session_id']  # append levels needed (e.g., trials, units)
        self.approaches = approaches or ['bootstrap', 'traditional', 'mixed-effects']  # 'mixed-effects', 'summary' as other option
        self.tests = tests or ['direct_prob', 'mann-whitney', 'emmeans', 'anova', 'spearmanr']  # 'wilcoxon' as other option
        self.alternatives = alternatives or ['two-sided']  # 'greater', 'less' as other options
        self.nboot = nboot  # number of iterations to perform for bootstrapping default should be 1000
        self.units = units or 'trials'  # lowest hierarchical level of individual samples to use for description
        self.results_type = results_type or 'preliminary'  # indicates which folder to save the data in
        if self.results_type == 'manuscript':
            folder_name = Path(__file__).parent.parent.parent / 'results' / 'manuscript_figures'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=folder_name)
        else:
            self.results_io = results_io or ResultsIO(creator_file=__file__, folder_name=Path(__file__).stem)

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

    def get_summary(self, df, dependent_vars=None, group_vars='group', filename=''):
        self.dependent_vars = dependent_vars
        self.group_vars = group_vars
        self._setup_data(approach='traditional', data=df)

        self.descript_df = self._get_descriptive_stats(approach='traditional', test='na', alternative='na')
        self.stats_df = pd.DataFrame()

        self._export_stats(filename, summary_only=True)

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
            self.df_processed = data.copy(deep=True)
            self.model = self._get_mixed_effects_model(data)
        else:
            warnings.warn(f'Statistical approach {approach} is not supported')
            self.df_processed = []

        return self

    def _perform_test(self, approach, test, alternative, pairs=None):
        pair_outputs = []

        # run tests that look across pairs
        for var in self.dependent_vars:
            if test == 'emmeans':
                emm = emmeans.emmeans(self.model, 'predictor', contr='pairwise', adjust='tukey')
                emm_df = self.r_to_pandas_df(ro.r['summary'](emm.rx2('contrasts')))
                emm_df['pairs'] = [[literal_eval(c) for c in row.split(' - ')] for row in emm_df['contrast'].to_list()]
                for _, row in emm_df.iterrows():
                    matching_pair = [p for p in pairs if (p[0] in row['pairs'] and p[1] in row['pairs'])]
                    if matching_pair:
                        test_statistic_name = [r for r in ['z.ratio', 't.ratio'] if r in row.index][0]
                        test_output = dict(pair=matching_pair, variable=var, test=test, approach=approach,
                                           test_statistic=row.get(test_statistic_name),
                                           test_statistic_name=test_statistic_name,
                                           df=row.get('df'), p_val=row["p.value"],
                                           alternative='two-sided')
                        pair_outputs.append(test_output)
            elif test == 'anova':
                if approach == 'mixed_effects':
                    anova_df = self.r_to_pandas_df(ro.r['as.data.frame'](ro.r['anova'](self.model, ddf="Kenward-Roger")))
                    test_output = dict(pair=((self.group_vars[0], ''),), variable=var, test=test, approach=approach,
                                       test_statistic=anova_df['F value'].to_numpy()[0],
                                       test_statistic_name='F value',
                                       df=anova_df['DenDF'], p_val=anova_df["Pr(>F)"].to_numpy()[0],
                                       alternative='anova with Kenward-Rogers method')
                    pair_outputs.append(test_output)
                else:
                    print('not currently supported')
                    # output = pg.mixed_anova()  # TODO - determine if I need mixed or not
            elif test == 'spearman':
                test_val, pval = spearmanr(self.df_processed[self.group_vars[0]], self.df_processed[var],
                                           alternative=alternative, nan_policy='omit')
                test_output = dict(pair=((self.group_vars[0], ''),), variable=var, test=test, approach=approach,
                                   test_statistic=test_val, test_statistic_name='rho',  # not exactly right? but using
                                   df=(len(self.df_processed[var]) - 2), p_val=pval,
                                   alternative=alternative)
                pair_outputs.append(test_output)

        # run tests that work on pairs individually
        if test in ['direct_prob', 'mann-whitney', 'wilcoxon', 'wilcoxon_one_sample']:
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
                        test_output = dict(pair=p, variable=var, test=test, approach=approach, test_statistic=[prob_vals],
                                           test_statistic_name='joint_probability', df=np.nan,
                                           p_val=np.min(prob_vals) * 2, alternative=comparisons[np.argmin(prob_vals)])
                    elif test == 'mann-whitney':
                        output = pg.mwu(samples[0][var].to_numpy(), samples[1][var].to_numpy(), alternative=alternative)
                        test_output = dict(pair=p, variable=var, test=test, approach=approach,
                                           test_statistic=output['U-val'].to_numpy()[0],
                                           test_statistic_name='U-val', df=np.nan,
                                           p_val=output["p-val"].to_numpy()[0],
                                           alternative=output['alternative'].to_numpy()[0])
                    elif test == 'wilcoxon':
                        output = pg.wilcoxon(samples[0][var].to_numpy(), samples[1][var].to_numpy(), alternative=alternative)
                        test_output = dict(pair=p, variable=var, test=test, approach=approach,
                                           test_statistic=output['W-val'].to_numpy()[0],
                                           test_statistic_name='W-val', df=np.nan,
                                           p_val=output["p-val"].to_numpy()[0],
                                           alternative=output['alternative'].to_numpy()[0])
                    elif test == 'wilcoxon_one_sample':
                        output = pg.wilcoxon(samples[0][var].to_numpy(), alternative=alternative)
                        test_output = dict(pair=p, variable=var, test=test, approach=approach,
                                           test_statistic=output['W-val'].to_numpy()[0],
                                           test_statistic_name='W-val', df=np.nan,
                                           p_val=output["p-val"].to_numpy()[0],
                                           alternative=output['alternative'].to_numpy()[0])
                        pair_outputs.append(test_output)

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
                              .reset_index()
                              .assign(sem=lambda x: x['std'] / np.sqrt(x['count'])))
        descriptive_stats.insert(0, 'approach', approach)
        descriptive_stats.insert(1, 'test', test)
        descriptive_stats.insert(2, 'alternative', alternative)

        if test == 'spearman':
            descriptive_stats[self.group_vars[0]] = self.group_vars[0]
            descriptive_stats = descriptive_stats.iloc[0:2, :]

        return descriptive_stats

    def _export_stats(self, filename, summary_only=False):
        self.results_io.export_statistics(self.descript_df, f'{filename}_descriptive', results_type=self.results_type,
                                          format='csv')
        self.results_io.export_statistics(self._get_stats_text(), f'{filename}_text', results_type=self.results_type,
                                          format='txt')

        if not summary_only:
            self.results_io.export_statistics(self.stats_df, f'{filename}_p_values', results_type=self.results_type,
                                              format='csv')

    def _get_stats_text(self):
        descript_text = self.descript_df.apply(lambda x: self.descript_to_text(x), axis=1)
        descript_text = ''.join(descript_text)

        stats_text = self.stats_df.apply(lambda x: self.stats_to_text(x), axis=1)
        stats_text = ''.join(stats_text)

        return ''.join([descript_text, stats_text])

    def descript_to_text(self, x):
        text = f'{", ".join([x[v] for v in self.group_vars])}: {x["mean"]:.2f} ± {x["sem"]:.2f}, ' \
               f'n = {x["count"]:.0f} {self.units}, ' \
               f'percentiles = {", ".join([f"{x:.2f}" for x in x.loc["min":"max"].to_list()])} {new_line}'
        return text

    @staticmethod
    def stats_to_text(x):
        comparison = f'{x["pair"][0][0]} vs. {x["pair"][0][1]}' if len(x['pair'][0]) > 1 else f'{x["pair"][0][0]}'
        if x['df'] is not np.nan:
            text = f'{x["test_statistic_name"]}({x["df"]}) = {x["test_statistic"]}, p = {x["p_val"]:.4f}, ' \
                   f'{x["variable"]} for {comparison}, {x["test"]} {new_line} '
        else:
            text = f'{x["test_statistic_name"]} = {x["test_statistic"]}, p = {x["p_val"]:.4f}, ' \
                   f'{x["variable"]} for {comparison}, {x["test"]} {new_line} '
        return text

    def _get_mixed_effects_model(self, data):
        data['predictor'] = data[self.group_vars].apply(tuple, axis=1)
        for var in self.dependent_vars:

            formula = f'{var} ~ predictor + (1|animal) + (1|animal:session_id)'  # equivalent to var ~ predictor + 1|animal/session_id
            model = lme4.lmer(formula, data=self.pandas_to_r_df(data))
            # TODO - add checks for assumptions of mixed effects models

            # extra outputs that might be good to know
            # model2 = ro.r['lm'](f'{var} ~ predictor', data=self.pandas_to_r_df(data))
            # print(ro.r['anova'](model, model2))  # just for curiosity to compare with/without random effects
            # print(report.report(model))
            # print(ro.r['summary'](model))  (could access variables within using rx2)

        return model

    @staticmethod
    def pandas_to_r_df(data):
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(data)

        return r_df

    @staticmethod
    def r_to_pandas_df(data):
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.rpy2py(data)

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
