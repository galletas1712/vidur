import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd

from vidur.config import DistributionShiftRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)

logger = logging.getLogger(__name__)


class DistributionShiftRequestLengthGenerator(BaseRequestLengthGenerator):
    """
    Token length generator that loads two trace files and creates a distribution shift pattern.
    It starts with one trace, shifts abruptly to the other trace in the middle, 
    then returns to the original trace.
    """

    def __init__(self, config: DistributionShiftRequestLengthGeneratorConfig):
        super().__init__(config)
        
        # Extract parameters from config
        self.config = config
        self.primary_trace_file = getattr(config, "primary_trace_file", "data/processed_traces/splitwise_conv.csv")
        self.stage2secondary_trace_file = getattr(config, "stage2secondary_trace_file", "data/processed_traces/splitwise_code.csv")
        self.distribution_shift_ratio = getattr(config, "distribution_shift_ratio", 0.3)
        self.prefill_scale_factor = getattr(config, "prefill_scale_factor", 1.0)
        self.decode_scale_factor = getattr(config, "decode_scale_factor", 1.0)
        
        # Load primary trace
        logger.info(f"Loading primary trace file: {self.primary_trace_file}")
        self.primary_trace_df = pd.read_csv(self.primary_trace_file)
        self._preprocess_trace_df(self.primary_trace_df, "primary")
        
        # Load stage2secondary trace
        logger.info(f"Loading stage2secondary trace file: {self.stage2secondary_trace_file}")
        self.stage2secondary_trace_df = pd.read_csv(self.stage2secondary_trace_file)
        self._preprocess_trace_df(self.stage2secondary_trace_df, "stage2secondary")

        # Set random seed
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Prepare the data for distribution shift pattern
        self._prepare_data()
        
    def _preprocess_trace_df(self, df: pd.DataFrame, trace_name: str):
        """Process and validate the dataframe, applying scaling factors."""
        # Scale prefill and decode tokens
        df["num_prefill_tokens"] = (
            df["num_prefill_tokens"] * self.prefill_scale_factor
        )
        df["num_decode_tokens"] = (
            df["num_decode_tokens"] * self.decode_scale_factor
        )

        # Make sure all the prefill and decode counts are integers
        df["num_prefill_tokens"] = df["num_prefill_tokens"].astype(int)
        df["num_decode_tokens"] = df["num_decode_tokens"].astype(int)

        # Make sure that there is at least one prefill and decode token
        df["num_prefill_tokens"] = df["num_prefill_tokens"].clip(lower=1)
        df["num_decode_tokens"] = df["num_decode_tokens"].clip(lower=1)

        # Make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = df["num_prefill_tokens"] + df["num_decode_tokens"]
        diff_tokens = total_tokens - self.config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        df["num_prefill_tokens"] = df["num_prefill_tokens"] - diff_tokens

        assert all(
            df["num_prefill_tokens"] + df["num_decode_tokens"]
            <= self.config.max_tokens
        )

        logger.info(
            f"Processed {trace_name} trace with {len(df)} requests"
        )
    
    def _prepare_data(self):
        """
        Prepare the data for distribution shift pattern.
        We sample from both traces and arrange the token sequences in the desired pattern.
        """
        # Determine the number of samples to take from each trace
        total_samples = self.config.num_requests
        stage2secondary_count = int(total_samples * self.distribution_shift_ratio)
        primary_count = total_samples - stage2secondary_count
        stage1primary_count = primary_count // 2
        stage3primary_count = primary_count - stage1primary_count
        
        logger.info(f"Creating distribution shift pattern:")
        logger.info(f"  Primary trace (first segment): {stage1primary_count} samples")
        logger.info(f"  stage2secondary trace (middle segment): {stage2secondary_count} samples")
        logger.info(f"  Primary trace (last segment): {stage3primary_count} samples")
        
        # Shuffle and sample from both traces
        primary_df = self.primary_trace_df.sample(frac=1, random_state=self.config.seed)
        stage2secondary_df = self.stage2secondary_trace_df.sample(frac=1, random_state=self.config.seed)
        
        # Create three segments
        self.all_samples = []
        
        # First segment from primary trace (looping if needed)
        for i in range(stage1primary_count):
            idx = i % len(primary_df)
            self.all_samples.append((
                primary_df.iloc[idx]["num_prefill_tokens"],
                primary_df.iloc[idx]["num_decode_tokens"]
            ))
        
        # Middle segment from stage2secondary trace (looping if needed)
        for i in range(stage2secondary_count):
            idx = i % len(stage2secondary_df)
            self.all_samples.append((
                stage2secondary_df.iloc[idx]["num_prefill_tokens"],
                stage2secondary_df.iloc[idx]["num_decode_tokens"]
            ))
        
        # Last segment from primary trace (looping if needed)
        for i in range(stage3primary_count):
            idx = (i + stage1primary_count) % len(primary_df)
            self.all_samples.append((
                primary_df.iloc[idx]["num_prefill_tokens"],
                primary_df.iloc[idx]["num_decode_tokens"]
            ))
        
        # Initialize counter
        self.current_idx = 0
    
    def get_next_num_tokens(self) -> Tuple[float, float]:
        """Get the next token counts following the distribution shift pattern."""
        if self.current_idx >= len(self.all_samples):
            return None, None
        
        result = self.all_samples[self.current_idx]
        self.current_idx += 1
        return result