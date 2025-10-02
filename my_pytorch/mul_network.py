import torch 

class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs,30),
            torch.nn.ReLU(),
            #第二个隐藏层
            torch.nn.Linear(30,20),
            torch.nn.ReLU(),
            # 输出层
            torch.nn.Linear(20,num_outputs),
        )
    def forward(self,x):
        logits = self.layers(x)
        return logits

if __name__ == "__main__":
    model = NeuralNetwork(50,3)
    print(model)
    
    num_params = sum( p.numel() for p in model.parameters() if p.requires_grad )
    print(f'Number of parameters: {num_params}')
    print(model.layers[0].weight.shape)
    
    torch.manual_seed(123)
    X = torch.rand(1,50)
    
    with torch.no_grad():
        out = model(X)
        out  = torch.softmax(out,dim=1)
    print(out)