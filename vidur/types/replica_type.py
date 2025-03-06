from vidur.types.base_int_enum import BaseIntEnum


class ReplicaType(BaseIntEnum):
    """
    Types of replicas that can be used in the cluster.
    
    PREFILL_ONLY: Replica that can only process prefill phase of requests
    DECODE_ONLY: Replica that can only process decode phase of requests
    HYBRID: Replica that can process both prefill and decode phases
    """
    PREFILL_ONLY = 0
    DECODE_ONLY = 1
    HYBRID = 2