from typing import Dict, List, Tuple

from vidur.entities import Request
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

logger = init_logger(__name__)


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            
            # Get eligible replicas based on request phase
            eligible_replicas = self.get_eligible_replicas_for_request(request)
            
            if not eligible_replicas:
                # Put the request back and stop scheduling since we can't handle it
                self._request_queue.insert(0, request)
                logger.warning(f"No eligible replicas for request {request.id} (is_prefill_complete={request.is_prefill_complete})")
                break
                
            # Find the eligible replica with the least outstanding requests
            eligible_pending_map = {
                replica_id: pending_requests_map[replica_id]
                for replica_id in eligible_replicas
            }
            
            replica_id = min(eligible_pending_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))

        return request_mapping