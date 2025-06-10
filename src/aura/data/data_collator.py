from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator for speech-to-sequence tasks that handles padding of input features
    and labels for batch processing.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Processes input features and labels for a batch, applying padding and 
        other necessary transformations.
        
        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): A list of dictionaries containing
                input features and labels.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing padded input features and labels.
        """
        # Process input features by extracting and padding them
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Process labels by extracting and padding them
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens with -100 to correctly ignore them during loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the beginning-of-sequence (BOS) token if it was added in tokenization
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add the processed labels back to the batch
        batch["labels"] = labels

        # Add the domain information to the batch
        if features[0].get("domain", None) is not None:
            batch["domain"] = torch.tensor([feature["domain"] for feature in features])

        return batch
