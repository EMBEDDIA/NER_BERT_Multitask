import torch

from torch import nn
from transformers import BertForTokenClassification, BertModel

from transformers.modeling_bert import BertOnlyMLMHead

from architectures.crf import allowed_transitions, ConditionalRandomField



class BERT_model(BertForTokenClassification):

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.use_crf = False
		self.predict_masked = False
		if hasattr(config, "task_specific_params"):
			other_config = config.task_specific_params
			if other_config["crf"]:
				self.use_crf = True
				crf_constraints = allowed_transitions(other_config["type_crf_constraints"], dict(map(reversed, config.label2id.items())))
				self.crf = ConditionalRandomField(config.num_labels, constraints=crf_constraints)
			if other_config["predict_masked"]:
				self.predict_masked = True
				self.cls = BertOnlyMLMHead(config)
		self.init_weights()

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, tokens_mask=None,
				labelling_mask=None, lm_mask=None, lm_labels=None, valid_output_tokens=None, valid_output_predict=None):

		original_sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)

		batch_size, max_len, feat_dim = original_sequence_output.shape

		if tokens_mask is not None:
			#valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
			for i in range(batch_size):
				jj = -1
				for j in range(max_len):
					if tokens_mask[i][j].item() == 1:
						jj += 1
						valid_output_tokens[i][jj] = original_sequence_output[i][j]
			linear_sequence_output = self.dropout(valid_output_tokens)
			del valid_output_tokens
		else:
			linear_sequence_output = self.dropout(original_sequence_output)
		logits = self.classifier(linear_sequence_output)

		loss_masks = 0.0
		if self.predict_masked and lm_labels is not None:
			#valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
			for i in range(batch_size):
				jj = -1
				for j in range(max_len):
					if lm_mask[i][j].item() == 1:
						jj += 1
						valid_output_predict[i][jj] = original_sequence_output[i][j]
			lm_sequence_output = self.dropout(valid_output_predict)
			del valid_output_predict

			lm_logits = self.cls(lm_sequence_output)

			loss_lm_fct = nn.CrossEntropyLoss(ignore_index=0)

			active_logits = lm_logits.view(-1, self.config.vocab_size)
			active_labels = lm_labels.view(-1)
			loss_masks = loss_lm_fct(active_logits, active_labels)

		del original_sequence_output

		loss = 0.0
		if self.use_crf:
			if labels is not None:
				loss = self.crf(logits, labels, labelling_mask)
				loss = loss.mean()
		else:
			if labels is not None:
				loss_fct = nn.CrossEntropyLoss(ignore_index=0)
				if labelling_mask is not None:
					active_tokens = labelling_mask.view(-1) == 1
					active_logits = logits.view(-1, self.num_labels)[active_tokens]
					active_labels = labels.view(-1)[active_tokens]
					loss = loss_fct(active_logits, active_labels)
				else:
					loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

		if labels is not None:
			if self.predict_masked:
				loss += loss_masks
			return loss, logits
		else:
			return logits

	def predictMasked(self):
		return self.predict_masked

	def hasCRF(self):
		return self.use_crf

	def getCRFtags(self, logits, labelling_mask):
		if self.use_crf:
			_, logits = self.crf.viterbi_tags(logits, labelling_mask)
			return logits
		else:
			return None

	#From https://github.com/huggingface/transformers/issues/1730#issuecomment-550081307
	def resize_embedding_and_fc(self, new_num_tokens):

		if self.predict_masked:
			# Change the FC
			old_fc = self.cls.predictions.decoder
			self.cls.predictions.decoder = self._get_resized_fc(old_fc, new_num_tokens)

			# Change the bias
			old_bias = self.cls.predictions.bias
			self.cls.predictions.bias = self._get_resized_bias(old_bias, new_num_tokens)

		# Change the embedding
		self.resize_token_embeddings(new_num_tokens)

	def _get_resized_bias(self, old_bias, new_num_tokens):
		old_num_tokens = old_bias.data.size()[0]
		if old_num_tokens == new_num_tokens:
			return old_bias

		# Create new biases
		new_bias = nn.Parameter(torch.zeros(new_num_tokens))
		new_bias.to(old_bias.device)

		# Copy from the previous weights
		num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
		new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
		return new_bias

	def _get_resized_fc(self, old_fc, new_num_tokens):

		old_num_tokens, old_embedding_dim = old_fc.weight.size()
		if old_num_tokens == new_num_tokens:
			return old_fc

		# Create new weights
		new_fc = nn.Linear(in_features=old_embedding_dim, out_features=new_num_tokens)
		new_fc.to(old_fc.weight.device)

		# initialize all weights (in particular added tokens)
		self._init_weights(new_fc)

		# Copy from the previous weights
		num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
		new_fc.weight.data[:num_tokens_to_copy, :] = old_fc.weight.data[:num_tokens_to_copy, :]
		return new_fc

