import os
import numpy as np
import pandas as pd
from simtools.Analysis.BaseAnalyzers import BaseAnalyzer

column_types = dict(Time='uint16',
                    time='uint16',
                    Run_Number='uint8',
                    Node='uint8',
                    node='uint8',
                    NodeID='uint8'
                    )


def optimize_dataframe(simdata):
    """
    Set the data to the optimal size for their types. For examples, we know this experiment does not have
    more than 255 nodes so we should assign the NodeId columns to an uint8 since it takes less memory than the
    default uint64
    :param simdata:
    :return:
    """
    for column, item_type in column_types.items():
        if column in simdata.columns:
            simdata[column] = simdata[column].astype(item_type)
    return simdata


class InsetChartAnalyzer(BaseAnalyzer):

    def __init__(self, exp_name, sweep_variables, working_dir='.'):
        super(InsetChartAnalyzer, self).__init__(
            working_dir=working_dir,
            # filenames=['output/ReportVectorGenetics_gambiae_ALLELE_FREQ.csv']
            filenames=['output/InsetChart.json']
        )
        self.exp_name = exp_name
        self.sweep_variables = sweep_variables
        self.channels = ['New Clinical Cases', 'Elimination', 'Annual EIR']
        self.output_fname = os.path.join(self.working_dir, "%s_clinical_cases" % self.exp_name)

    def select_simulation_data(self, data, simulation):
        simdata = []

        datatemp = data[self.filenames[0]]

        # prevalence = datatemp['Channels']['PfHRP2 Prevalence']['Data']
        true_prevalence = datatemp['Channels']['True Prevalence']['Data']
        incidence = datatemp['Channels']['New Clinical Cases']['Data'][180:6 * 365]
        annual_eir = sum(datatemp['Channels']['Daily EIR']['Data']) / 8
        elimination = 0
        if sum(true_prevalence[7*365:8*365]) == 0:
            elimination = 1
        datatemp = pd.DataFrame(list(zip([sum(incidence)], [elimination], [annual_eir])),
                            columns=['New Clinical Cases', 'Elimination', 'Annual EIR'])

        simdata.append(datatemp)
        simdata = pd.concat(simdata)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                simdata[sweep_var] = simulation.tags[sweep_var]
            else:
                simdata[sweep_var] = 0

        simdata.reset_index(drop=True)
        return optimize_dataframe(simdata)

    def finalize(self, d: pd.DataFrame, partition_vars: list, partition_vars_vals: list):

        fname_suffix = ''
        for ipv, partition_var in enumerate(partition_vars):
            fname_suffix = fname_suffix + '_' + partition_var + str(partition_vars_vals[ipv])

        d_mean = d.groupby(self.sweep_variables)[self.channels].apply(
            np.mean).reset_index()
        d_std = d.groupby(self.sweep_variables)[self.channels].apply(
            np.std).reset_index()
        for channel in self.channels:
            d_mean[channel + '_std'] = d_std[channel]
        d_mean.to_csv(self.output_fname + fname_suffix + '.csv', index=False)