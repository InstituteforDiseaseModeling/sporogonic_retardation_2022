import os
import sys

sys.path.append(os.path.dirname(__file__))

from partitioning_analyzer import PartitioningDataAnalyzeManager

from allele_frequency_analyzer import AlleleFreqAnalyzer
from inset_chart_analyzer import InsetChartAnalyzer
from prevalence_time_series_analyzer import PrevalenceAnalyzer


if __name__ == "__main__":

    experiments = {
        "experiment_name":
            "expi_id",
    }

    sweep_vars = ['Larval_Capacity', 'Transmission_To_Human', 'Infected_Progress']

    for expt_name, exp_id in experiments.items():
        am = PartitioningDataAnalyzeManager(exp_list=exp_id,
                                            partitionable_columns=sweep_vars,
                                            analyzers=
                                            [
                                                AlleleFreqAnalyzer(
                                                    exp_name=expt_name,
                                                    sweep_variables=sweep_vars
                                                ),
                                                InsetChartAnalyzer(
                                                    exp_name=expt_name,
                                                    sweep_variables=sweep_vars
                                                ),
                                                PrevalenceAnalyzer(
                                                    exp_name=expt_name,
                                                    sweep_variables=sweep_vars
                                                )
                                            ]
                                            )
        print(am.experiments)
        am.analyze()