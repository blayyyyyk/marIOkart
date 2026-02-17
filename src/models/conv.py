import torch.nn as nn
import torch

class ConvModel(nn.Module):
    def __init__(self, obs_size, out_channels, stride, kernel_size, dilation, embedding_size, n_layers):
        super(ConvModel, self).__init__()
        assert n_layers > 0, "must have at least one conv layer"
        self.action_embed = nn.Embedding(out_channels, embedding_size)
        
        in_channels = obs_size + embedding_size
        self.conv_seq = nn.Sequential()
        
        self.conv_seq.append(nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation
        ))
        
        for i in range(n_layers-1):
            conv = nn.Conv1d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride,
                dilation=dilation
            )
            self.conv_seq.append(conv)
        
        self.proj = nn.Linear(out_channels, out_channels)
        
        
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        
        obs, act_ids = x[:, :, :-1], x[:, :, -1]
        action_embed = self.action_embed(act_ids.to(torch.int)) # -> (B, T, embedding_size)
        features = torch.cat([obs, action_embed], dim=-1) # -> (B, T, obs_dim+embedding_size-1)
        features = features.permute(0, 2, 1)
        
        conv = self.conv_seq(features)
        conv = conv.permute(0, 2, 1)
        
        logits = self.proj(conv)
        return logits
        
        
def main():
    model = ConvModel(
        obs_size=13, 
        out_channels=11, 
        stride=1, 
        kernel_size=3, 
        dilation=1, 
        embedding_size=8, 
        n_layers=1
    )
    x1 = torch.rand(20, 23, 13)
    x2 = torch.randint(0, 8, (20, 23, 1), dtype=torch.float)
    x = torch.cat([x1, x2], dim=-1)
    y = model(x)
    print(x.shape, y.shape)
    
    
if __name__ == "__main__":
    main()
        
        