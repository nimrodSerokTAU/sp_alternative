from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
from typing import Literal, List, Any, Iterator, Tuple, Optional
import math
class BatchGenerator(Sequence):
    def __init__(self, features, true_labels, true_msa_ids, train_msa_ids, val_msa_ids, aligners, batch_size, validation_split=0.2, is_validation=False, repeats=1, mixed_portion=0.0, per_aligner=False, classification_task = False, features_w_names=np.nan, only_intra_msa: bool = True):
        self.only_intra_msa = only_intra_msa
        self.features = features
        self.true_labels = np.asarray(true_labels)
        self.msa_ids = true_msa_ids  # TRUE MSA IDs (categorical)
        self.batch_size = batch_size
        self.unique_msa_ids = np.unique(true_msa_ids)
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.val_msa_ids = val_msa_ids
        self.train_msa_ids = train_msa_ids
        self.repeats = repeats
        self.mixed_portion = mixed_portion
        self.per_aligner = per_aligner
        self.aligners = aligners
        self.unique_aligners = np.unique(aligners)[np.unique(aligners) != "true"]
        # self.features_w_names = features_w_names
        self.features_w_names = features_w_names.reset_index(drop=True)

        self.classification_task = classification_task

        if self.is_validation:
            mask = np.isin(self.msa_ids, self.val_msa_ids)

        else:
            mask = np.isin(self.msa_ids, self.train_msa_ids)

        self.features = self.features[mask]
        self.true_labels = self.true_labels[mask]
        self.features_w_names = self.features_w_names[mask]
        self.features_w_names = self.features_w_names.reset_index(drop=True)
        self.msa_ids = self.msa_ids[mask]
        self.unique_msa_ids = np.unique(self.msa_ids)
        self.batches = self._precompute_batches()

    def _split_idx_into_batches(self, idx: np.ndarray) -> Tuple[List[Any], List[Any]]:
        batches: List[Any] = []
        remaining_samples_set: List[Any] = []
        # np.random.shuffle(idx)
        num_samples = len(idx)
        num_full_batches = num_samples // self.batch_size
        remaining_samples = num_samples % self.batch_size
        leaving_out = math.floor(self.mixed_portion * num_full_batches)

        for i in range(num_full_batches - leaving_out): # I want to leave out some batches into the mix of remaining samples
            batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
            batches.append((self.features[batch_idx], self.true_labels[batch_idx]))

        if remaining_samples > 0 or leaving_out > 0: # intermixed batches (consisting of the samples from different unique MSA IDs) to make sure that
            remaining_samples_set.extend(idx[(num_full_batches - leaving_out) * self.batch_size:])
        np.random.shuffle(remaining_samples_set)
        np.random.shuffle(batches)
        return batches, remaining_samples_set


    def _precompute_batches(self) -> List[Any]:
        batches: List[Any] = []
        batches_mix: List[Any] = []
        remaining_samples_set: List[Any] = []

        for msa_id in self.unique_msa_ids:
            try:
                for k in range(self.repeats): #testing an option to produce different batch mixes
                    idx = np.where(self.msa_ids == msa_id)[0]
                    if len(idx) > self.batch_size:
                        if self.per_aligner:
                            for aligner in self.unique_aligners:
                                idx_aln = np.intersect1d(np.where(self.aligners == aligner)[0], idx)
                                if len(idx_aln) > self.batch_size:
                                    np.random.shuffle(idx_aln) #TODO check that shuffling here instead of within _split_idx_into_batches doesn't mess with the results
                                    btchs, rem_sam_set = self._split_idx_into_batches(idx_aln)
                                    batches.extend(btchs)
                                    remaining_samples_set.extend(rem_sam_set)
                                else:
                                    continue
                        else:
                            np.random.shuffle(idx) #TODO check that shuffling here instead of within _split_idx_into_batches doesn't mess with the results
                            btchs, rem_sam_set = self._split_idx_into_batches(idx)
                            batches.extend(btchs)
                            remaining_samples_set.extend(rem_sam_set)

            except Exception as e:
                print(f"Exception {e}\n")

        remaining_samples_set = np.array(remaining_samples_set)
        np.random.shuffle(remaining_samples_set)

        for i in range(0, len(remaining_samples_set), self.batch_size):
            batch_idx = remaining_samples_set[i: i + self.batch_size]
            if len(batch_idx) == self.batch_size:
                batches_mix.append((self.features[batch_idx], self.true_labels[batch_idx]))

        if self.only_intra_msa:
            final_batches = batches
        else:
            final_batches = batches + batches_mix

        # final_batches = batches + batches_mix
        np.random.shuffle(final_batches)

        return final_batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> Any:
        return self.batches[idx]

    def on_epoch_end(self) -> None:
        if not self.is_validation:
            self.batches = self._precompute_batches()
        np.random.shuffle(self.batches)

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for idx in range(len(self)):
            batch_features, batch_labels = self[idx]
            yield (batch_features, batch_labels)