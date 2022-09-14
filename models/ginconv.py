import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim*2)
        #self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        # cell line feature

        self.conv_xt_1 = nn.Conv1d(in_channels=735, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        #self.conv_xt_4 = nn.Conv1d(in_channels=n_filters*3, out_channels=4 * n_filters, kernel_size=8)
        #self.pool_xt_4 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(128 * 6, 1024)
        self.fc2_xt = nn.Linear(1024, output_dim)
        '''
        self.conv_xt_1 = nn.LSTM(256,embed_dim, num_layers=2, batch_first = True,bidirectional=True)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.LSTM(85, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.LSTM(85, embed_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(85, 256)
        self.fc2_xt = nn.Linear(256, output_dim)
        '''
        
        #smiles
        self.embedding_xds = nn.Embedding(num_embeddings=65, embedding_dim=128*2)
        self.conv_xds_1 = nn.Conv1d(in_channels=100, out_channels=25,kernel_size=8)
        self.conv_xds_2 = nn.Conv1d(in_channels=25, out_channels=50, kernel_size=8)
        self.conv_xds_3 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=8)
        #self.conv_xds_4 = nn.Conv1d(in_channels=75, out_channels=100, kernel_size=8)
        self.fc1_xds = nn.Linear(100 * 6, 1024)
        self.fc2_xds = nn.Linear(1024, 128)
        
        # combined layers
        #self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(x)
        #print(data.target)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # protein input feed-forward:
        target = data.target   #[1024, 735]
        target = target.long()
        target = self.embedding_xt(target)
    
        #print(f"target.shape {target.shape}")
        # 1d conv layers
        conv_xt = self.conv_xt_1(target)
        # conv_xt = conv_xt[0]
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        #print(f"conv_xt1.shape {conv_xt.shape}")

        conv_xt = self.conv_xt_2(conv_xt)
        # conv_xt = conv_xt[0]
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        #print(f"conv_xt2.shape {conv_xt.shape}")
        
        conv_xt = self.conv_xt_3(conv_xt)
        # conv_xt = conv_xt[0]
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
    
        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = torch.relu(self.fc1_xt(xt))
        xt = F.dropout(xt, p=0.2, training=self.training)
        xt = self.fc2_xt(xt)
        xt = F.dropout(xt, p=0.2, training=self.training)
        
        drug = data.drug
        #print("======drug.shape===========")
        #print(drug.shape)
        drug = drug.long()
        embedded_xds = self.embedding_xds(drug) 
        conv_xds = self.conv_xds_1(embedded_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.pool_xt_1(conv_xds)

        conv_xds = self.conv_xds_2(conv_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.pool_xt_2(conv_xds)

        conv_xds = self.conv_xds_3(conv_xds)
        conv_xds = torch.relu(conv_xds)
        conv_xds = self.pool_xt_3(conv_xds)
        conv_xds = conv_xds.view(-1, conv_xds.shape[1] * conv_xds.shape[2])

        drug = torch.relu(self.fc1_xds(conv_xds))
        drug = F.dropout(drug, p=0.2, training=self.training) 
        drug = self.fc2_xds(drug)
        drug = F.dropout(drug, p=0.2, training=self.training) 
        # concat
        #xc = torch.cat((x, xt), 1)
        xc = torch.cat((x, drug, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        #out = nn.Sigmoid()(out)
        return out, x
