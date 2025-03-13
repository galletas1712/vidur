from typing import List, Tuple

from vidur.entities import Request
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Maintain separate counters for prefill and decode phases
        self._prefill_counter = 0
        self._decode_counter = 0

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            
            # Get eligible replicas based on request phase
            eligible_replicas = list(self.get_eligible_replicas_for_request(request))
            
            if not eligible_replicas:
                # Put the request back and stop scheduling since we can't handle it
                self._request_queue.insert(0, request)
                logger.warning(f"No eligible replicas for request {request.id} (is_prefill_complete={request.is_prefill_complete})")
                break
                
            # Use appropriate counter based on request phase
            if request.is_prefill_complete:
                # Decode phase - use decode counter
                replica_index = self._decode_counter % len(eligible_replicas)
                self._decode_counter += 1
            else:
                # Prefill phase - use prefill counter
                replica_index = self._prefill_counter % len(eligible_replicas)
                self._prefill_counter += 1
            
            replica_id = eligible_replicas[replica_index]
            logger.debug(f"RR Global Scheduler assigning request {request.id} to replica {replica_id}")
            request_mapping.append((replica_id, request))

        return request_mapping