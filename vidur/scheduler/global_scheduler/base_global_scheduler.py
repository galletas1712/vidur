from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.logger import init_logger
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)

logger = init_logger(__name__)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)
        
        # Create maps of replica IDs by type for quick access
        self._prefill_capable_replicas = {
            replica_id for replica_id, replica in replicas.items() 
            if replica.can_handle_prefill
        }
        self._decode_capable_replicas = {
            replica_id for replica_id, replica in replicas.items() 
            if replica.can_handle_decode
        }
        
        # Print out information about replica types
        logger.info(f"Prefill-capable replicas: {self._prefill_capable_replicas}")
        logger.info(f"Decode-capable replicas: {self._decode_capable_replicas}")

        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_configs[replica_id].get_type(),
                replica_config=config.cluster_config.replica_configs[replica_id],
                replica_scheduler_config=config.cluster_config.replica_scheduler_configs[replica_id],
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=ExecutionTimePredictorRegistry.get(
                    config.execution_time_predictor_config.get_type(),
                    predictor_config=config.execution_time_predictor_config,
                    replica_config=config.cluster_config.replica_configs[replica_id],
                    replica_scheduler_config=config.cluster_config.replica_scheduler_configs[replica_id],
                    metrics_config=config.metrics_config,
                ),
            )
            for replica_id, replica in replicas.items()
        }
        self._request_queue = []

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    def get_eligible_replicas_for_request(self, request: Request) -> Set[int]:
        """
        Get the set of replicas that can handle the given request based on its phase.
        
        Args:
            request: The request to find eligible replicas for
            
        Returns:
            A set of replica IDs that can handle the request
        """
        if request.is_prefill_complete:
            # If the request has completed prefill, it needs decode-capable replicas
            return self._decode_capable_replicas
        else:
            # If the request is in prefill phase, it needs prefill-capable replicas
            return self._prefill_capable_replicas
    
    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass