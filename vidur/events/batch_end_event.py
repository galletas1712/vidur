from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

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

        # The batch's on_batch_end method updates the processed tokens for all requests
        self._batch.on_batch_end(self.time)
        
        # Get the current replica scheduler
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        
        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        # Let the replica scheduler handle the batch end event for remaining requests
        # If the replica cannot handle decode, this will free just-finished prefills
        requests_to_migrate = replica_scheduler.on_batch_end(self._batch)
        next_events = []
        if requests_to_migrate:
            for request in requests_to_migrate:
                scheduler.add_request(request)
            next_events.append(GlobalScheduleEvent(self.time))
        
        # Schedule this replica to potentially form new batches
        next_events.append(ReplicaScheduleEvent(self.time, self._replica_id))
        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }