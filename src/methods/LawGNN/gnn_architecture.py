import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


# GCN architecture for Bi-Encoder
class GCNBiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, method='cosine'):
        super(GCNBiEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method

        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'cosine':
            cos_sim = torch.nn.functional.cosine_similarity(x[a_idx], x[b_idx], dim=1)
            cos_prob = (cos_sim + 1) / 2  # Map from [-1, 1] to [0, 1]
            logits = cos_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'cosine':
                # Adjust labels from {0,1} to {-1,1}
                labels_for_loss = labels.float() * 2 - 1
                loss_fct = torch.nn.CosineEmbeddingLoss()
                loss = loss_fct(x[a_idx], x[b_idx], labels_for_loss)
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# GraphSAGE architecture for Bi-Encoder
class SAGEBiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, method='cosine'):
        super(SAGEBiEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method
        
        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'cosine':
            cos_sim = torch.nn.functional.cosine_similarity(x[a_idx], x[b_idx], dim=1)
            cos_prob = (cos_sim + 1) / 2  # Map from [-1, 1] to [0, 1]
            logits = cos_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'cosine':
                # Adjust labels from {0,1} to {-1,1}
                labels_for_loss = labels.float() * 2 - 1
                loss_fct = torch.nn.CosineEmbeddingLoss()
                loss = loss_fct(x[a_idx], x[b_idx], labels_for_loss)
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
# GATv2 architecture for Bi-Encoder
class GATv2BiEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, method='cosine'):
        super(GATv2BiEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)
        self.method = method
        
        
        if self.method == 'linear':
            self.classifier = torch.nn.Linear(out_channels * 2, 1)  # For concatenated representations

    def forward(self, x, edge_index, a_idx, b_idx, labels=None):
        # Forward pass for article A
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Compute similarity
        if self.method == 'cosine':
            cos_sim = torch.nn.functional.cosine_similarity(x[a_idx], x[b_idx], dim=1)
            cos_prob = (cos_sim + 1) / 2  # Map from [-1, 1] to [0, 1]
            logits = cos_prob.unsqueeze(-1)  # Shape [batch_size, 1]
        elif self.method == 'linear':
            combined_output = torch.cat([x[a_idx], x[b_idx]], dim=1)  # Concatenate representations
            logits = self.classifier(combined_output)  # Shape [batch_size, 1]

        loss = None
        if labels is not None:
            if self.method == 'cosine':
                # Adjust labels from {0,1} to {-1,1}
                labels_for_loss = labels.float() * 2 - 1
                loss_fct = torch.nn.CosineEmbeddingLoss()
                loss = loss_fct(x[a_idx], x[b_idx], labels_for_loss)
            elif self.method == 'linear':
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float().unsqueeze(-1))

        return logits, loss
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


from src.utils.encoder.crossencoder_utils import CrossEncoderModel
from src.utils.encoder.utils import ThreeLayerClassifier

class NoGNNCrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(NoGNNCrossEncoderModel, self).__init__(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 2 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.vector_tensor

        # Encoder output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        # Combine the CLS token embedding with the article embeddings
        # print(pooled_output.shape, x[article1_idx].shape, x[article2_idx].shape)
        combined_output = torch.cat([pooled_output, x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


class GCNCrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(GCNCrossEncoderModel, self).__init__(model_name)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 2 * out_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        combined_output = torch.cat([
            pooled_output,
            x[article1_idx],
            x[article2_idx]
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


class SAGECrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(SAGECrossEncoderModel, self).__init__(model_name)
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 2 * out_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        combined_output = torch.cat([
            pooled_output,
            x[article1_idx],
            x[article2_idx]
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
class GATv2CrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor,heads=8, method='cosine'):
        super(GATv2CrossEncoderModel, self).__init__(model_name)
        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)

        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 2 * out_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        combined_output = torch.cat([
            pooled_output,
            x[article1_idx],
            x[article2_idx]
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


class VanillaSAGECrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor):
        super(VanillaSAGECrossEncoderModel, self).__init__(model_name)
        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 4 * out_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        combined_output = torch.cat([
            pooled_output,
            self.vector_tensor[article1_idx],
            self.vector_tensor[article2_idx],
            x[article1_idx],
            x[article2_idx]
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)    

class VanillaGATv2CrossEncoderModel(CrossEncoderModel):
    def __init__(self, model_name, in_channels, out_channels, vector_tensor, edge_index_tensor,heads=8, method='cosine'):
        super(VanillaGATv2CrossEncoderModel, self).__init__(model_name)
        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)

        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = self.encoder.config.hidden_size + 4 * out_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, input_ids=None, attention_mask=None, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] if self.encoder.config.model_type in ["bert", "big_bird"] else outputs.last_hidden_state[:, -1, :]  # BERT는 [CLS], GPT는 마지막 토큰 사용
        pooled_output = self.dropout(pooled_output)

        combined_output = torch.cat([
            pooled_output,
            self.vector_tensor[article1_idx],
            self.vector_tensor[article2_idx],
            x[article1_idx],
            x[article2_idx]
        ], dim=1)
        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


# crossencoder 조차 없는 경우
class NoCrossVanillaEncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels,vector_tensor, edge_index_tensor):
        super(NoCrossVanillaEncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        # Calculate combined input size
        combined_input_size = 2 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.vector_tensor

        combined_output = torch.cat([x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

class NoCrossSAGEEncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels,vector_tensor, edge_index_tensor):
        super(NoCrossSAGEEncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

        # Calculate combined input size
        combined_input_size = 2 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        combined_output = torch.cat([x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
    
# crossencoder 조차 없는 경우
class NoCrossGCNEncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels,vector_tensor, edge_index_tensor):
        super(NoCrossGCNEncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

        # Calculate combined input size
        combined_input_size = 2 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        # print(pooled_output.shape, x[article1_idx].shape, x[article2_idx].shape)
        combined_output = torch.cat([x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

class NoCrossGATv2EncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels,vector_tensor, edge_index_tensor, heads=8):
        super(NoCrossGATv2EncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vector_tensor', vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)

        # Calculate combined input size
        combined_input_size = 2 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.conv1(self.vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        # print(pooled_output.shape, x[article1_idx].shape, x[article2_idx].shape)
        combined_output = torch.cat([x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

# vanilla-gat hybrid method
class NoCrossVanillaGATv2EncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, vanilla_vector_tensor, gnn_vector_tensor, edge_index_tensor, heads=8):
        super(NoCrossVanillaGATv2EncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vanilla_vector_tensor', vanilla_vector_tensor)
        self.register_buffer('gnn_vector_tensor', gnn_vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        self.conv1 = GATv2Conv(in_channels, 16, heads=heads)
        self.conv2 = GATv2Conv(16 * heads, out_channels, heads=1)
        self.dropout = torch.nn.Dropout(0.1)

        # Calculate combined input size
        combined_input_size = 4 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.conv1(self.gnn_vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        combined_output = torch.cat([self.vanilla_vector_tensor[article1_idx], self.vanilla_vector_tensor[article2_idx], x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
    

class NoCrossVanillaSAGEEncoderModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, vanilla_vector_tensor, gnn_vector_tensor, edge_index_tensor, heads=8):
        super(NoCrossVanillaSAGEEncoderModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        self.register_buffer('vanilla_vector_tensor', vanilla_vector_tensor)
        self.register_buffer('gnn_vector_tensor', gnn_vector_tensor)
        self.register_buffer('edge_index_tensor', edge_index_tensor)

        self.conv1 = SAGEConv(in_channels, 16)
        self.conv2 = SAGEConv(16, out_channels)
        self.dropout = torch.nn.Dropout(0.1)

        # Calculate combined input size
        combined_input_size = 4 * in_channels

        # Override the classifier with the correct input size
        self.classifier = torch.nn.Linear(combined_input_size, 1)
        # self.classifier = ThreeLayerClassifier(combined_input_size, 1)

    def forward(self, article1_idx, article2_idx, labels=None):
        x = self.conv1(self.gnn_vector_tensor, self.edge_index_tensor)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index_tensor)

        combined_output = torch.cat([self.vanilla_vector_tensor[article1_idx], self.vanilla_vector_tensor[article2_idx], x[article1_idx], x[article2_idx]], dim=1)
        logits = self.classifier(combined_output)  # Shape: [batch_size, 1]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Reshape logits and labels to ensure consistent sizes
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)