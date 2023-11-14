from typing import Optional, List
from transformers import FuyuPreTrainedModel, FuyuConfig, AutoModelForCausalLM
from transformers.models.fuyu.modeling_fuyu import FuyuVisionEmbedTokens
import torch

class PatchPrediction(torch.nn.Linear):
    pass

class FuyuWithPatchPrediction(FuyuPreTrainedModel):
    def __init__(self, config: FuyuConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if getattr(config, "_flash_attn_2_enabled", False):
            config.text_config._flash_attn_2_enabled = True

        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        self.vision_embed_tokens = FuyuVisionEmbedTokens(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
        )

        self.next_patch_predictor = PatchPrediction(
            config.hidden_size,
            config.patch_size * config.patch_size * config.num_channels,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
            continuous_embeddings:
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
            shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
            indices in image_patch_input_indices for that batch element.
            image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_patches: Optional[torch.Tensor] = None,
        image_patches_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(
                        patch.to(self.vision_embed_tokens.weight.dtype)
                    )
                    .squeeze(0)
                    .to(inputs_embeds.dtype)
                    for patch in image_patches
                ]
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            labels=labels,
            use_cache=use_cache,
        )

        hidden_states = outputs.hidden_states[-1]
        patch_predictions = self.next_patch_predictor(hidden_states)
        return outputs, patch_predictions

    def get_patch_prediction_loss(self, batch, patch_predictions):
        # > 0 makes sure that we skip the first element of the sequence
        # (note that >= 0 includes all elements)
        # This is like shifting labels in causal language modeling but
        # accounts for batching correctly
        patch_predictions = patch_predictions[
            batch["image_patches_indices"] > 0
        ]
        targets = torch.concat(
            [
                image_patches[:, 1:, :]
                for image_patches in batch["image_patches"]
            ],
            dim=1,
        ).squeeze()
        criterion = torch.nn.MSELoss()
        mse_loss = criterion(
            patch_predictions, targets.to(patch_predictions.dtype)
        )
        return mse_loss

