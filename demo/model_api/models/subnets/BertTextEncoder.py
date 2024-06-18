import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

__all__ = ['BertTextEncoder']

class BertTextEncoder(nn.Module):
    def __init__(self, finetune=False, pretrained='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)
        self.finetune = finetune
    
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
