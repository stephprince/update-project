from pynwb import NWBFile

from update_project.base_analysis_class import BaseAnalysisClass
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.decoding.bayesian_decoder_analyzer import BayesianDecoderAnalyzer


class ExampleTrialAnalyzer(BaseAnalysisClass):
    def __init__(self, nwbfile: NWBFile, session_id: str, params=dict(), feature=None, regions=None):
        self.session_data = []
        self.session_id = session_id
        self.nwbfile = self._setup_data(nwbfile)

        self.feature = feature or 'y_position'
        self.regions = regions or [['CA1'], ['PFC']]
        self.both_regions_only = params.get('both_regions_only', True)
        self.exclusion_criteria = params.get('exclusion_criteria', dict(units=20, trials=50))
        self.single_unit_params = params.get('single_unit_params', dict(align_window=15,
                                                                        align_times=['t_update']))
        self.decoding_params = params.get('decoding_params', dict(encoder_bin_num=50,
                                                                  decoder_bin_size=0.2,
                                                                  decoder_test_size=0.2))

    def _setup_data(self, nwbfile: NWBFile):
        return nwbfile

    def run_analysis(self, overwrite=True):
        print(f'Analyzing behavioral data for {self.session_id}...')

        for reg in self.regions:
            self.decoding_params.update(units_types=dict(region=reg,
                                                         cell_type=['Pyramidal Cell', 'Narrow Interneuron',
                                                                    'Wide Interneuron']))
            self.single_unit_params.update(units_types=dict(region=reg,
                                                            cell_type=['Pyramidal Cell', 'Narrow Interneuron',
                                                                       'Wide Interneuron']))

            # load existing data
            single_unit = SingleUnitAnalyzer(nwbfile=self.nwbfile, session_id=self.session_id,
                                          feature=self.feature,
                                          params=self.single_unit_params)
            single_unit.run_analysis(overwrite=overwrite,
                                  export_data=False)  # don't use existing data but also don't save it bc example only

            decoder = BayesianDecoderAnalyzer(nwbfile=self.nwbfile, session_id=self.session_id,
                                              features=[self.feature],
                                              params=self.decoding_params)
            decoder.run_analysis(export_data=False)

            # save to group output
            session_data = dict(single_unit=single_unit,
                                decoder=decoder,
                                region=reg)
            self.session_data.append(session_data)



