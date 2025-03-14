import os
from functools import reduce
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly_express as px
import wandb

from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.logger import init_logger
from vidur.metrics.cdf_sketch import CDFSketch
from vidur.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CpuOperationMetrics,
    DistributionShiftStage,
    OperationMetrics,
    RequestCompletionMetricsTimeSeries,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenCompletionMetricsTimeSeries,
    TokenMetricsTimeDistribution,
)
from vidur.metrics.data_series import DataSeries
from vidur.metrics.series_average_meter import SeriesAverageMeter
from vidur.types.request_length_generator_type import RequestLengthGeneratorType
from vidur.utils.mfu_calculator import MFUCalculator

logger = init_logger(__name__)


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"
STAGE_STR = "Distribution Shift Stage"


class MetricsStore:

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self._simulation_config = simulation_config
        self._config = self._simulation_config.metrics_config
        self._last_request_arrived_at = None

        # copy config
        self._num_replicas = self._simulation_config.cluster_config.num_replicas
        self._num_pipeline_stages = [
            replica_config.num_pipeline_stages
            for replica_config in self._simulation_config.cluster_config.replica_configs
        ]

        # Check if using DistributionShiftRequestLengthGenerator
        self._is_distribution_shift = (
            hasattr(self._simulation_config.request_generator_config, "length_generator_config") and
            self._simulation_config.request_generator_config.length_generator_config.get_type() == RequestLengthGeneratorType.DISTRIBUTION_SHIFT
        )
        self._distribution_stage_boundaries = None
        if self._is_distribution_shift:
            # Calculate distribution shift boundaries based on configuration
            self._calculate_distribution_shift_boundaries()
        
        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage metrics for distribution shift if applicable
        self._per_stage_request_metrics: Dict[
            DistributionShiftStage, Dict[RequestMetricsTimeDistributions, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_request_metrics[stage] = {}
                for metric_name in RequestMetricsTimeDistributions:
                    self._per_stage_request_metrics[stage][metric_name] = DataSeries(
                        REQUEST_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, CDFSketch
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage token metrics for distribution shift if applicable
        self._per_stage_token_metrics: Dict[
            DistributionShiftStage, Dict[TokenMetricsTimeDistribution, CDFSketch]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_token_metrics[stage] = {}
                for metric_name in TokenMetricsTimeDistribution:
                    self._per_stage_token_metrics[stage][metric_name] = CDFSketch(
                        f"{metric_name.value}",
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage histogram metrics for distribution shift if applicable
        self._per_stage_request_histogram: Dict[
            DistributionShiftStage, Dict[RequestMetricsHistogram, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_request_histogram[stage] = {}
                for metric_name in RequestMetricsHistogram:
                    self._per_stage_request_histogram[stage][metric_name] = DataSeries(
                        REQUEST_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        # Initialise batch metrics
        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, CDFSketch
        ] = {}
        self._batch_metrics_count_distribution_per_batch: Dict[
            BatchMetricsCountDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_count_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage batch count metrics for distribution shift if applicable
        self._per_stage_batch_count: Dict[
            DistributionShiftStage, Dict[BatchMetricsCountDistribution, CDFSketch]
        ] = {}
        self._per_stage_batch_count_per_batch: Dict[
            DistributionShiftStage, Dict[BatchMetricsCountDistribution, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_batch_count[stage] = {}
                self._per_stage_batch_count_per_batch[stage] = {}
                for metric_name in BatchMetricsCountDistribution:
                    self._per_stage_batch_count[stage][metric_name] = CDFSketch(
                        f"{metric_name.value}",
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )
                    self._per_stage_batch_count_per_batch[stage][metric_name] = DataSeries(
                        BATCH_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, CDFSketch
        ] = {}
        self._batch_metrics_time_distribution_per_batch: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_time_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage batch time metrics for distribution shift if applicable
        self._per_stage_batch_time: Dict[
            DistributionShiftStage, Dict[BatchMetricsTimeDistribution, CDFSketch]
        ] = {}
        self._per_stage_batch_time_per_batch: Dict[
            DistributionShiftStage, Dict[BatchMetricsTimeDistribution, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_batch_time[stage] = {}
                self._per_stage_batch_time_per_batch[stage] = {}
                for metric_name in BatchMetricsTimeDistribution:
                    self._per_stage_batch_time[stage][metric_name] = CDFSketch(
                        f"{metric_name.value}",
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )
                    self._per_stage_batch_time_per_batch[stage][metric_name] = DataSeries(
                        BATCH_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        # Initialise completion metrics
        self._request_completion_metrics_time_series: Dict[
            RequestCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestCompletionMetricsTimeSeries:
            self._request_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage request completion metrics for distribution shift if applicable
        self._per_stage_request_completion: Dict[
            DistributionShiftStage, Dict[RequestCompletionMetricsTimeSeries, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_request_completion[stage] = {}
                for metric_name in RequestCompletionMetricsTimeSeries:
                    self._per_stage_request_completion[stage][metric_name] = DataSeries(
                        TIME_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )
                    
        self._token_completion_metrics_time_series: Dict[
            TokenCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in TokenCompletionMetricsTimeSeries:
            self._token_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage token completion metrics for distribution shift if applicable
        self._per_stage_token_completion: Dict[
            DistributionShiftStage, Dict[TokenCompletionMetricsTimeSeries, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_token_completion[stage] = {}
                for metric_name in TokenCompletionMetricsTimeSeries:
                    self._per_stage_token_completion[stage][metric_name] = DataSeries(
                        TIME_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        # Initialise operation metrics
        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self._operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage operation metrics for distribution shift if applicable
        self._per_stage_operation_metrics: Dict[
            DistributionShiftStage, Dict[OperationMetrics, CDFSketch]
        ] = {}
        self._per_stage_operation_metrics_per_batch: Dict[
            DistributionShiftStage, Dict[OperationMetrics, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_operation_metrics[stage] = {}
                self._per_stage_operation_metrics_per_batch[stage] = {}
                for metric_name in OperationMetrics:
                    self._per_stage_operation_metrics[stage][metric_name] = CDFSketch(
                        f"{metric_name.value}",
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )
                    self._per_stage_operation_metrics_per_batch[stage][metric_name] = DataSeries(
                        BATCH_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        self._cpu_operation_metrics_per_batch: Dict[CpuOperationMetrics, DataSeries] = (
            {}
        )
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._cpu_operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Per-stage CPU operation metrics for distribution shift if applicable
        self._per_stage_cpu_operation_metrics: Dict[
            DistributionShiftStage, Dict[CpuOperationMetrics, CDFSketch]
        ] = {}
        self._per_stage_cpu_operation_metrics_per_batch: Dict[
            DistributionShiftStage, Dict[CpuOperationMetrics, DataSeries]
        ] = {}
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                self._per_stage_cpu_operation_metrics[stage] = {}
                self._per_stage_cpu_operation_metrics_per_batch[stage] = {}
                for metric_name in CpuOperationMetrics:
                    self._per_stage_cpu_operation_metrics[stage][metric_name] = CDFSketch(
                        f"{metric_name.value}",
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )
                    self._per_stage_cpu_operation_metrics_per_batch[stage][metric_name] = DataSeries(
                        BATCH_ID_STR,
                        f"{metric_name.value}",
                        self._config.subsamples,
                        self._config.save_table_to_wandb,
                        self._config.store_plots,
                    )

        # per replica metrics
        self._replica_memory_usage = []
        # per replica stage metrics
        self._replica_busy_time = []
        self._replica_mfu = []
        self._mfu_calculators = {}
        for replica_idx, replica_config in enumerate(simulation_config.cluster_config.replica_configs):
            self._mfu_calculators[replica_idx] = MFUCalculator(replica_config)

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage.append(
                SeriesAverageMeter(
                    TIME_STR,
                    MEMORY_USAGE_STR,
                    self._config.save_table_to_wandb,
                )
            )
            self._replica_memory_usage[replica_idx].put(0, 0)

            self._replica_busy_time.append([])
            self._replica_mfu.append([])

            for stage_idx in range(self._num_pipeline_stages[replica_idx]):
                self._replica_busy_time[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        BUSY_TIME_PERCENT,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )
                self._replica_busy_time[replica_idx][stage_idx].put(0, 0)

                self._replica_mfu[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        UTILIZATION_STR,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )
                self._replica_mfu[replica_idx][stage_idx].put(0, 0)

        self._init_wandb()
        
        # Add dictionary to track last batch completion time for each request
        self._last_batch_completion_time: Dict[int, float] = {}

    def _calculate_distribution_shift_boundaries(self) -> None:
        """Calculate the request boundaries for distribution shift stages."""
        if not hasattr(self._simulation_config.request_generator_config, "num_requests"):
            # If we don't have the necessary config, don't set up distribution shift tracking
            self._is_distribution_shift = False
            return
            
        if not hasattr(self._simulation_config.request_generator_config, "distribution_shift_ratio"):
            # If ratio isn't specified, use the default value
            distribution_shift_ratio = 0.3
        else:
            distribution_shift_ratio = self._simulation_config.request_generator_config.distribution_shift_ratio
            
        total_requests = self._simulation_config.request_generator_config.num_requests
        stage2secondary_count = int(total_requests * distribution_shift_ratio)
        primary_count = total_requests - stage2secondary_count
        stage1primary_count = primary_count // 2
        
        # Store the request index ranges for each stage
        self._distribution_stage_boundaries = {
            DistributionShiftStage.stage1primary: (0, stage1primary_count),
            DistributionShiftStage.stage2secondary: (stage1primary_count, stage1primary_count + stage2secondary_count),
            DistributionShiftStage.stage3primary: (stage1primary_count + stage2secondary_count, total_requests)
        }
        logger.info(f"Distribution shift stages: {self._distribution_stage_boundaries}")

    def _get_current_stage(self, request_id: int) -> Optional[DistributionShiftStage]:
        """Determine the current distribution shift stage based on request ID."""
        if not self._is_distribution_shift or self._distribution_stage_boundaries is None:
            return None
            
        for stage, (start, end) in self._distribution_stage_boundaries.items():
            if start <= request_id < end:
                return stage
                
        return None

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._config.wandb_project
            or not self._config.wandb_group
        ):
            return

        wandb.init(
            project=self._config.wandb_project,
            group=self._config.wandb_group,
            name=self._config.wandb_run_name,
            config=self._simulation_config.to_dict(),
        )

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)

        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on=[key_to_join], how="outer"),
            [dataseries._to_df() for dataseries in dataseries_list],
        )
        merged_df.to_csv(f"{base_path}/{file_name}.csv", index=False)
        if wandb.run and self._config.save_table_to_wandb:
            wand_table = wandb.Table(dataframe=merged_df)
            wandb.log({f"{file_name}_table": wand_table}, step=0)

    def _store_bar_plot(
        self,
        base_path: str,
        plot_name: str,
        x_label: str,
        y_label: str,
        data: Dict[str, float],
    ):
        if wandb.run:
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )
        if self._config.store_plots:
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                labels={"x": x_label, "y": y_label},
            )
            fig.write_image(f"{base_path}/{plot_name}.png")

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:
            return

        total_operation_runtimes: Dict[str, float] = {}

        total_operation_runtimes["model_execution_e2e"] = 0
        for dataseries in self._operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum
            total_operation_runtimes["model_execution_e2e"] += dataseries.sum

        for dataseries in self._cpu_operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum

        self._store_bar_plot(
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,
            TIME_STR_MS,
            total_operation_runtimes,
        )

        # Store per-stage operation metrics if using distribution shift
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                stage_total_operation_runtimes: Dict[str, float] = {}
                stage_total_operation_runtimes["model_execution_e2e"] = 0
                
                for metric_name, dataseries in self._per_stage_operation_metrics[stage].items():
                    dataseries.plot_cdf(
                        base_plot_path, 
                        f"{stage.value}_{metric_name.value}_execution_time", 
                        TIME_STR_MS
                    )
                    stage_total_operation_runtimes[metric_name.value] = dataseries.sum
                    stage_total_operation_runtimes["model_execution_e2e"] += dataseries.sum

                for metric_name, dataseries in self._per_stage_cpu_operation_metrics[stage].items():
                    dataseries.plot_cdf(
                        base_plot_path, 
                        f"{stage.value}_{metric_name.value}_execution_time", 
                        TIME_STR_MS
                    )
                    stage_total_operation_runtimes[metric_name.value] = dataseries.sum

                self._store_bar_plot(
                    base_plot_path,
                    f"{stage.value}_total_operation_runtimes",
                    OPERATION_STR,
                    TIME_STR_MS,
                    stage_total_operation_runtimes,
                )

        if not self._config.keep_individual_batch_metrics:
            return

        for dataseries in self._operation_metrics_per_batch.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        operations_dataseries_list = list(self._operation_metrics_per_batch.values())
        self._save_as_csv(
            dataseries_list=operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="operation_metrics",
        )

        for dataseries in self._cpu_operation_metrics_per_batch.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        cpu_operations_dataseries_list = list(
            self._cpu_operation_metrics_per_batch.values()
        )
        self._save_as_csv(
            dataseries_list=cpu_operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="cpu_operation_metrics",
        )

        # Store per-stage per-batch operation metrics if using distribution shift
        if self._is_distribution_shift and self._config.keep_individual_batch_metrics:
            for stage in DistributionShiftStage:
                for dataseries in self._per_stage_operation_metrics_per_batch[stage].values():
                    dataseries.consolidate()
                    dataseries.plot_step(
                        base_plot_path,
                        f"{stage.value}_{dataseries._metric_name}_per_batch",
                        y_axis_label=TIME_STR_MS,
                        y_cumsum=False,
                    )
                operations_dataseries_list = list(self._per_stage_operation_metrics_per_batch[stage].values())
                if operations_dataseries_list:  # Only save if we have data
                    self._save_as_csv(
                        dataseries_list=operations_dataseries_list,
                        key_to_join=BATCH_ID_STR,
                        base_path=self._config.output_dir,
                        file_name=f"{stage.value}_operation_metrics",
                    )

                for dataseries in self._per_stage_cpu_operation_metrics_per_batch[stage].values():
                    dataseries.consolidate()
                    dataseries.plot_step(
                        base_plot_path,
                        f"{stage.value}_{dataseries._metric_name}_per_batch",
                        y_axis_label=TIME_STR_MS,
                        y_cumsum=False,
                    )
                cpu_operations_dataseries_list = list(
                    self._per_stage_cpu_operation_metrics_per_batch[stage].values()
                )
                if cpu_operations_dataseries_list:  # Only save if we have data
                    self._save_as_csv(
                        dataseries_list=cpu_operations_dataseries_list,
                        key_to_join=BATCH_ID_STR,
                        base_path=self._config.output_dir,
                        file_name=f"{stage.value}_cpu_operation_metrics",
                    )

    def _store_request_metrics(self, base_plot_path: str):
        if not self._config.store_request_metrics:
            return

        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())

        self._save_as_csv(
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        for dataseries in self._request_metrics_histogram.values():
            dataseries.plot_histogram(base_plot_path, dataseries._y_name)

        for dataseries in self._request_metrics_time_distributions.values():
            dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)

        # Store per-stage request metrics if using distribution shift
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                stage_metrics = list(
                    self._per_stage_request_metrics[stage].values()
                ) + list(self._per_stage_request_histogram[stage].values())
                
                # Only save if we have data
                if any(len(ds) > 0 for ds in stage_metrics):
                    self._save_as_csv(
                        dataseries_list=stage_metrics,
                        key_to_join=REQUEST_ID_STR,
                        base_path=self._config.output_dir,
                        file_name=f"{stage.value}_request_metrics",
                    )

                for dataseries in self._per_stage_request_histogram[stage].values():
                    if len(dataseries) > 0:  # Only plot if we have data
                        dataseries.plot_histogram(base_plot_path, f"{stage.value}_{dataseries._y_name}")

                for dataseries in self._per_stage_request_metrics[stage].values():
                    if len(dataseries) > 0:  # Only plot if we have data
                        dataseries.plot_cdf(base_plot_path, f"{stage.value}_{dataseries._y_name}", TIME_STR)

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        for dataseries in self._batch_metrics_time_distribution.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, y_axis_label)

        for dataseries in self._batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)

        # Store per-stage batch metrics if using distribution shift
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                for metric_name, dataseries in self._per_stage_batch_time[stage].items():
                    y_axis_label = (
                        TIME_STR_MS
                        if "model_execution" in dataseries._metric_name
                        else TIME_STR
                    )
                    dataseries.plot_cdf(base_plot_path, dataseries._metric_name, y_axis_label)

                for dataseries in self._per_stage_batch_count[stage].values():
                    dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)

        if not self._config.keep_individual_batch_metrics:
            return

        for dataseries in self._batch_metrics_time_distribution_per_batch.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=y_axis_label,
                y_cumsum=False,
            ),

        for dataseries in self._batch_metrics_count_distribution_per_batch.values():
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=COUNT_STR,
                y_cumsum=False,
            ),

        all_batch_metrics = list(
            self._batch_metrics_count_distribution_per_batch.values()
        ) + list(self._batch_metrics_time_distribution_per_batch.values())

        self._save_as_csv(
            dataseries_list=all_batch_metrics,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="batch_metrics",
        )

        # Store per-stage per-batch metrics if using distribution shift
        if self._is_distribution_shift and self._config.keep_individual_batch_metrics:
            for stage in DistributionShiftStage:
                for dataseries in self._per_stage_batch_time_per_batch[stage].values():
                    if len(dataseries) == 0:
                        continue
                    y_axis_label = (
                        TIME_STR_MS
                        if "model_execution" in dataseries._metric_name
                        else TIME_STR
                    )
                    dataseries.plot_step(
                        base_plot_path,
                        f"{stage.value}_{dataseries._metric_name}_per_batch",
                        y_axis_label=y_axis_label,
                        y_cumsum=False,
                    )

                for dataseries in self._per_stage_batch_count_per_batch[stage].values():
                    if len(dataseries) == 0:
                        continue
                    dataseries.plot_step(
                        base_plot_path,
                        f"{stage.value}_{dataseries._metric_name}_per_batch",
                        y_axis_label=COUNT_STR,
                        y_cumsum=False,
                    )

                stage_batch_metrics = list(
                    self._per_stage_batch_count_per_batch[stage].values()
                ) + list(self._per_stage_batch_time_per_batch[stage].values())
                
                # Only save if we have data
                if any(len(ds) > 0 for ds in stage_batch_metrics):
                    self._save_as_csv(
                        dataseries_list=stage_batch_metrics,
                        key_to_join=BATCH_ID_STR,
                        base_path=self._config.output_dir,
                        file_name=f"{stage.value}_batch_metrics",
                    )

    def _store_completion_metrics(self, base_plot_path: str):
        if self._config.store_request_metrics:
            for dataseries in self._request_completion_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

            # Store per-stage completion metrics if using distribution shift
            if self._is_distribution_shift:
                for stage in DistributionShiftStage:
                    for dataseries in self._per_stage_request_completion[stage].values():
                        if len(dataseries) > 0:  # Only plot if we have data
                            dataseries.plot_step(
                                base_plot_path, f"{stage.value}_{dataseries._y_name}_time_series", COUNT_STR
                            )

        if not self._config.store_token_completion_metrics:
            return

        for dataseries in self._token_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, TIME_STR)

        for dataseries in self._token_completion_metrics_time_series.values():
            dataseries.plot_step(
                base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
            )
        
        logger.info(f"Token metrics: {self._token_metrics_time_distribution}")
        logger.info(f"Is distribution shift: {self._is_distribution_shift}")

        # Store per-stage token metrics if using distribution shift
        if self._is_distribution_shift:
            for stage in DistributionShiftStage:
                logger.info(f"Per stage token metrics: {self._per_stage_token_metrics[stage]}")
                for dataseries in self._per_stage_token_metrics[stage].values():
                    dataseries.plot_cdf(base_plot_path, f"{stage.value}_{dataseries._metric_name}", TIME_STR)

                for dataseries in self._per_stage_token_completion[stage].values():
                    if len(dataseries) > 0:  # Only plot if we have data
                        dataseries.plot_step(
                            base_plot_path, f"{stage.value}_{dataseries._y_name}_time_series", COUNT_STR
                        )

    def _store_utilization_metrics(self, base_plot_path: str):
        if not self._config.store_utilization_metrics:
            return

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage[replica_idx].print_stats(
                f"replica_{replica_idx + 1}_memory_usage", base_plot_path
            )
            for stage_idx in range(self._num_pipeline_stages[replica_idx]):
                self._replica_busy_time[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_busy_time_percent",
                    base_plot_path,
                )
                self._replica_mfu[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_mfu",
                    base_plot_path,
                )

    @if_write_metrics
    def plot(self) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)
        self._store_completion_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics(dir_plot_path)

    @if_write_metrics
    def on_request_arrival(self, time: float, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        # Determine the distribution shift stage if applicable
        current_stage = self._get_current_stage(request.id) if self._is_distribution_shift else None

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put(time, 1)

        # Track per-stage metrics if using distribution shift
        if current_stage is not None:
            self._per_stage_request_completion[current_stage][
                RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
            ].put(time, 1)

        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )

        # Track per-stage histogram metrics if using distribution shift
        if current_stage is not None:
            self._per_stage_request_histogram[current_stage][
                RequestMetricsHistogram.REQUEST_NUM_TOKENS
            ].put(request.id, request.total_tokens)
            self._per_stage_request_histogram[current_stage][
                RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
            ].put(request.id, request.num_prefill_tokens)
            self._per_stage_request_histogram[current_stage][
                RequestMetricsHistogram.REQUEST_DECODE_TOKENS
            ].put(request.id, request.num_decode_tokens)
            self._per_stage_request_histogram[current_stage][
                RequestMetricsHistogram.REQUEST_PD_RATIO
            ].put(request.id, request.pd_ratio)

        if self._last_request_arrived_at is not None:
            self._request_metrics_histogram[
                RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(request.id, request.arrived_at - self._last_request_arrived_at)
            
            # Track per-stage inter-arrival delay if using distribution shift
            if current_stage is not None:
                self._per_stage_request_histogram[current_stage][
                    RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
                ].put(request.id, request.arrived_at - self._last_request_arrived_at)
                
        self._last_request_arrived_at = request.arrived_at

    @if_write_metrics
    def _on_request_end(self, time: float, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        # Determine the distribution shift stage if applicable
        current_stage = self._get_current_stage(request.id) if self._is_distribution_shift else None

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)

        # Track per-stage metrics if using distribution shift
        if current_stage is not None:
            self._per_stage_request_completion[current_stage][
                RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
            ].put(request.completed_at, 1)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            (request.execution_time + request.preempted_time)
            / request.num_decode_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,
        )

        # Add the new metrics
        prefill_e2e_time = request.prefill_completed_at - request.arrived_at
        decode_e2e_time = request.completed_at - request.prefill_completed_at

        # 1. Prefill e2e time normalized
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E_NORMALIZED
        ].put(
            request.id,
            prefill_e2e_time / request.num_prefill_tokens if request.num_prefill_tokens > 0 else 0
        )
        
        # 2. Decode e2e time (unnormalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_E2E
        ].put(request.id, decode_e2e_time)
        
        # Decode e2e time (normalized) - using the existing field
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_E2E_NORMALIZED
        ].put(
            request.id,
            decode_e2e_time / request.num_decode_tokens if request.num_decode_tokens > 0 else 0
        )

        # Handle per-stage metrics if applicable
        if current_stage is not None:
            # 1. Prefill e2e time normalized
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.PREFILL_TIME_E2E_NORMALIZED
            ].put(
                request.id,
                prefill_e2e_time / request.num_prefill_tokens if request.num_prefill_tokens > 0 else 0
            )
            
            # 2. Decode e2e time (unnormalized)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.DECODE_TIME_E2E
            ].put(request.id, decode_e2e_time)
            
            # Decode e2e time (normalized)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.DECODE_TIME_E2E_NORMALIZED
            ].put(
                request.id,
                decode_e2e_time / request.num_decode_tokens if request.num_decode_tokens > 0 else 0
            )

        # Track per-stage distributions if using distribution shift
        if current_stage is not None:
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_E2E_TIME
            ].put(request.id, request.e2e_time)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
            ].put(request.id, request.e2e_time_normalized)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
            ].put(request.id, request.execution_time)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
            ].put(request.id, request.execution_time_normalized)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
            ].put(request.id, request.model_execution_time)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
            ].put(request.id, request.model_execution_time_normalized)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
            ].put(request.id, request.preempted_time)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
            ].put(request.id, request.scheduling_delay)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
            ].put(request.id, request.execution_time + request.preempted_time)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
            ].put(
                request.id,
                (request.execution_time + request.preempted_time)
                / request.num_decode_tokens,
            )
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.PREFILL_TIME_E2E
            ].put(request.id, request.prefill_completed_at - request.arrived_at)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
            ].put(request.id, request.prefill_completed_at - request.scheduled_at)
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
            ].put(
                request.id,
                (request.prefill_completed_at - request.scheduled_at)
                / request.num_prefill_tokens,
            )
            self._per_stage_request_metrics[current_stage][
                RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
            ].put(
                request.id,
                (request.completed_at - request.prefill_completed_at)
                / request.num_decode_tokens,
            )

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)

        # Track per-stage restart histogram if using distribution shift
        if current_stage is not None:
            self._per_stage_request_histogram[current_stage][
                RequestMetricsHistogram.REQUEST_NUM_RESTARTS
            ].put(request.id, request.num_restarts)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        # Determine the distribution shift stage if applicable
        current_stage = self._get_current_stage(request.id) if self._is_distribution_shift else None

        # if prefill has just finished in this iteration, update the prefill completion time series
        if (
            time == request.prefill_completed_at
            and self._config.store_token_completion_metrics
        ):
            self._token_completion_metrics_time_series[
                TokenCompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,
            )
            
            # Track per-stage metrics if using distribution shift
            if current_stage is not None:
                self._per_stage_token_completion[current_stage][
                    TokenCompletionMetricsTimeSeries.PREFILL_COMPLETIONS
                ].put(
                    time,
                    request.num_prefill_tokens,
                )

        # determine if this was prefill or decode token
        if not request.has_started_decode:
            return

        if not self._config.store_token_completion_metrics:
            return

        token_execution_time = time - batch.scheduled_at + request.latest_iteration_scheduling_delay
        
        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(token_execution_time)

        # Track per-stage token metrics if using distribution shift
        if current_stage is not None:
            self._per_stage_token_metrics[current_stage][
                TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
            ].put(token_execution_time)

        self._token_completion_metrics_time_series[
            TokenCompletionMetricsTimeSeries.DECODE_COMPLETIONS
        ].put(time, 1)

        # Track per-stage token completion metrics if using distribution shift
        if current_stage is not None:
            self._per_stage_token_completion[current_stage][
                TokenCompletionMetricsTimeSeries.DECODE_COMPLETIONS
            ].put(time, 1)

    def _push_metric(
        self, metric_name: OperationMetrics, batch_id: int, value: float, 
        stage: Optional[DistributionShiftStage] = None
    ) -> None:
        # Push to the regular metrics
        if metric_name in OperationMetrics:
            self._operation_metrics[metric_name].put(value)
            self._operation_metrics_per_batch[metric_name].put(batch_id, value)
            # Push to per-stage metrics if applicable
            if stage is not None and self._is_distribution_shift:
                self._per_stage_operation_metrics[stage][metric_name].put(value)
                self._per_stage_operation_metrics_per_batch[stage][metric_name].put(batch_id, value)
        elif metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name].put(value)
            self._cpu_operation_metrics_per_batch[metric_name].put(batch_id, value)
            # Push to per-stage metrics if applicable
            if stage is not None and self._is_distribution_shift:
                self._per_stage_cpu_operation_metrics[stage][metric_name].put(value)
                self._per_stage_cpu_operation_metrics_per_batch[stage][metric_name].put(batch_id, value)
        elif metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name].put(value)
            self._batch_metrics_time_distribution_per_batch[metric_name].put(
                batch_id, value
            )
            # Push to per-stage metrics if applicable
            if stage is not None and self._is_distribution_shift:
                self._per_stage_batch_time[stage][metric_name].put(value)
                self._per_stage_batch_time_per_batch[stage][metric_name].put(batch_id, value)
        elif metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name].put(value)
            self._batch_metrics_count_distribution_per_batch[metric_name].put(
                batch_id, value
            )
            # Push to per-stage metrics if applicable
            if stage is not None and self._is_distribution_shift:
                self._per_stage_batch_count[stage][metric_name].put(value)
                self._per_stage_batch_count_per_batch[stage][metric_name].put(batch_id, value)
        else:
            raise ValueError(f"Invalid metric name {metric_name}")

    @if_write_metrics
    def on_batch_end(
        self, time: float, batch: Batch, replica_id: int, memory_usage_percent: int
    ) -> None:
        if (
            self._config.min_batch_index and batch.id < self._config.min_batch_index
        ) or (self._config.max_batch_index and batch.id > self._config.max_batch_index):
            return

        # Determine the current distribution shift stage if applicable
        # For batches, use the first request's stage
        current_stage = None
        if self._is_distribution_shift and batch.requests:
            current_stage = self._get_current_stage(batch.requests[0].id)
            
        # Calculate time between tokens for each request in the batch
        for request in batch.requests:
            # Only track this for decode tokens (not prefill)
            if request.has_started_decode:
                if request.id in self._last_batch_completion_time:
                    tbt = time - self._last_batch_completion_time[request.id]
                    # 3. Time between tokens
                    self._token_metrics_time_distribution[
                        TokenMetricsTimeDistribution.TIME_BETWEEN_TOKENS
                    ].put(tbt)
                    
                    # Per-stage time between tokens if applicable
                    if current_stage is not None:
                        self._per_stage_token_metrics[current_stage][
                            TokenMetricsTimeDistribution.TIME_BETWEEN_TOKENS
                        ].put(tbt)
                
                # Update the last batch completion time for this request
                self._last_batch_completion_time[request.id] = time

        for request in batch.completed_requests:
            self._on_request_end(time, request)
            # Clean up the dictionary for completed requests
            if request.id in self._last_batch_completion_time:
                del self._last_batch_completion_time[request.id]

        if self._config.store_utilization_metrics:
            self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

        for request in batch.requests:
            self._update_per_token_execution_times(time, request, batch)

        if not self._config.store_batch_metrics:
            return

        self._push_metric(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME,
            batch.id,
            time - batch.scheduled_at,
            current_stage,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS,
            batch.id,
            batch.total_num_tokens,
            current_stage,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS,
            batch.id,
            batch.num_prefill_tokens,
            current_stage,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS,
            batch.id,
            batch.num_decode_tokens,
            current_stage,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_SIZE, batch.id, batch.size,
            current_stage,
        )

    @if_write_metrics
    def on_replica_schedule(
        self, time: float, replica_id: int, memory_usage_percent: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return

        self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:
        # NOTE: Don't collect per-stage operation metrics
        if not self._config.store_utilization_metrics:
            return

        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 100)
        mfu = self._mfu_calculators[replica_id].get_mfu(batch_stage)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, mfu)

        if not self._config.store_operation_metrics:
            return

        batch_id = batch_stage._batch_id
        for _ in range(execution_time.num_layers):
            self._push_metric(
                OperationMetrics.MLP_UP_PROJ,
                batch_id,
                execution_time.mlp_layer_up_proj_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_ACTIVATION,
                batch_id,
                execution_time.mlp_layer_act_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ,
                batch_id,
                execution_time.mlp_layer_down_proj_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.mlp_all_reduce_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_PRE_PROJ,
                batch_id,
                execution_time.attention_pre_proj_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ,
                batch_id,
                execution_time.attention_post_proj_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.attention_all_reduce_time,
            )

            if execution_time.attention_prefill_execution_time != 0:
                self._push_metric(
                    OperationMetrics.ATTN_PREFILL,
                    batch_id,
                    execution_time.attention_prefill_execution_time,
                )

            if execution_time.attention_decode_execution_time != 0:
                self._push_metric(
                    OperationMetrics.ATTN_DECODE,
                    batch_id,
                    execution_time.attention_decode_execution_time,
                )
            self._push_metric(
                OperationMetrics.ATTN_KV_CACHE_SAVE,
                batch_id,
                execution_time.attention_kv_cache_save_execution_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_ROPE,
                batch_id,
                execution_time.attention_rope_execution_time,
            )
            self._push_metric(
                OperationMetrics.ADD, batch_id, execution_time.add_time * 2,
            )
            self._push_metric(
                OperationMetrics.INPUT_LAYERNORM,
                batch_id,
                execution_time.attn_norm_time,
            )
            self._push_metric(
                OperationMetrics.POST_ATTENTION_LAYERNORM,
                batch_id,
                execution_time.mlp_norm_time,
            )

        self._push_metric(
            OperationMetrics.PIPELINE_SEND_RECV,
            batch_id,
            execution_time.pipeline_parallel_communication_time,
        )
        self._push_metric(
            CpuOperationMetrics.SCHEDULE, batch_id, execution_time.schedule_time,
        )
        self._push_metric(
            CpuOperationMetrics.SAMPLER_E2E, batch_id, execution_time.sampler_e2e_time,
        )
        self._push_metric(
            CpuOperationMetrics.PREPARE_INPUTS_E2E,
            batch_id,
            execution_time.prepare_inputs_e2e_time,
        )
        self._push_metric(
            CpuOperationMetrics.MODEL_EXECUTION_E2E,
            batch_id,
            execution_time.model_time_ms,
        )
        self._push_metric(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            batch_id,
            execution_time.process_model_outputs_time,
        )
        self._push_metric(
            CpuOperationMetrics.RAY_COMM_TIME, batch_id, execution_time.ray_comm_time,
        )

    @if_write_metrics
    def on_batch_stage_end(
        self, batch_stage: BatchStage, time: float, replica_id: int, stage_id: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 0)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, 0)