from random import choice
from typing import List, Tuple

from vidur.entities import Request
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)


class RandomGlobalScheduler(BaseGlobalScheduler):
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
                
            # Randomly select from eligible replicas
            replica_id = choice(eligible_replicas)
            request_mapping.append((replica_id, request))
            
        return request_mapping