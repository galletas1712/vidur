from abc import ABC, abstractmethod
from math import ceil
from typing import List

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        self._num_stages = num_stages

        self._replica_type = replica.replica_type
        self.can_handle_prefill = replica.can_handle_prefill
        self.can_handle_decode = replica.can_handle_decode

        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        memory_planner = MemoryPlanner(self._replica_config, replica)

        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
            and (
                not hasattr(self, "_preempted_requests")
                or len(self._preempted_requests) == 0
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        return request.num_prefill_tokens

    def add_request(self, request: Request) -> None:
        logger.debug(f"Adding request {request.id} to replica {self._replica_id}")
        self._request_queue.append(request)
        
    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    def free(self, *request_ids: List[int]) -> None:
        if hasattr(self, "_preempted_requests"):
            assert all(request_id not in self._preempted_requests for request_id in request_ids)
        for request_id in request_ids:
            if request_id not in self._allocation_map:
                continue
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                max(request.num_prefill_tokens, request.num_processed_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                 max(request.num_prefill_tokens, request.num_processed_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required <= 1
        ), f"num_tokens_required: {num_tokens_required} for request {request.id}, num_tokens_reserved: {num_tokens_reserved}, num_prefill_tokens: {request.num_prefill_tokens}, num_processed_tokens: {request.num_processed_tokens}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, num_tokens_required)

    def on_batch_end(self, batch: Batch) -> List[Request]:
        self._num_running_batches -= 1
        requests_to_migrate = []

        for request in batch.requests:
            if request.completed or (not self.can_handle_decode and request.just_finished_prefill):
                if request.completed:
                    logger.debug(f"Request {request.id} completed at replica {self._replica_id}: processed {request.num_processed_tokens}, prefill {request.num_prefill_tokens}, decode {request.num_decode_tokens}. Removing...")
                else:
                    requests_to_migrate.append(request)
                    logger.debug(f"Request {request.id} prefill completed at replica {self._replica_id}. : processed {request.num_processed_tokens}, prefill {request.num_prefill_tokens}, decode {request.num_decode_tokens}. Removing...")

                if request.id in self._allocation_map:
                    self.free(request.id)
            else:
                if not self.can_handle_decode:
                    assert not request.is_prefil_complete
                if not self.can_handle_prefill:
                    assert request.has_started_decode, f"Request {request.id} completion {request.completed}, worker type {self._replica_type}, processed {request.num_processed_tokens}, prefill {request.num_prefill_tokens}, decode {request.num_decode_tokens}"
                self._preempted_requests.append(request)
        
        return requests_to_migrate

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    def on_schedule(self) -> List[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches