from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_type import ReplicaType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        # Identify requests that just completed their prefill stage
        just_prefilled_requests = []
        
        # The batch's on_batch_end method updates the processed tokens for all requests
        self._batch.on_batch_end(self.time)
        
        # Check for requests that just completed prefill stage
        for request in self._batch.requests:
            # If a request just completed prefill (prefill_completed_at equals current time),
            # then we'll relocate it to a potentially different replica
            if request.prefill_completed_at == self.time:
                just_prefilled_requests.append(request)
                logger.debug(f"Request {request.id} completed prefill stage at {self.time}")
        
        # Get the current replica scheduler
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        
        # Remove the requests that just completed prefill from this replica
        # This also frees the memory allocated for these requests on this replica

        # TODO: Fix moving around requests when we have a hybrid replica leads to doubling for some reason
        if replica_scheduler._replica_type == ReplicaType.PREFILL_ONLY:
            for request in just_prefilled_requests:
                replica_scheduler.remove_request(request.id)
        
        # Let the replica scheduler handle the batch end event for remaining requests
        replica_scheduler.on_batch_end(self._batch)
        
        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )
        
        next_events = []
        
        # Schedule this replica to potentially form new batches
        next_events.append(ReplicaScheduleEvent(self.time, self._replica_id))
        
        # If we have requests that just completed prefill, add them to the global scheduler
        # for reassignment to potentially different replicas
        if just_prefilled_requests and replica_scheduler._replica_type == ReplicaType.PREFILL_ONLY:
            for request in just_prefilled_requests:
                scheduler.add_request(request)
            next_events.append(GlobalScheduleEvent(self.time))
            
        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }