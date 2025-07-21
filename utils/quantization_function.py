import torch
import torch.nn as nn

class WSQConv2d(nn.Module):
    bit1 = [-0.7979, 0.7979]
    bit2 = [-1.224, 0, 0.7646, 1.7242]	
    bit4 = [-2.6536, -1.9735, -1.508, -1.149, -0.8337, -0.5439, -0.2686, 0.,
            0.2303, 0.4648, 0.7081, 0.9663, 1.2481, 1.5676, 1.9679, 2.6488]


    def __init__(self, n_bits=1, clip_prob=-1):
        super(WSQConv2d, self).__init__()
        
        q_values = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        self.clip_prob = clip_prob
    
    def forward(self, x, global_x, std):
        with torch.no_grad():
            x = x - global_x    # residual
            
            # clip: V11
            if self.clip_prob > 0:
                x_abs = torch.abs(x)
                k = int((1 - self.clip_prob) * x_abs.numel())
                clip_threshold = torch.kthvalue(x_abs.view(-1), k).values
                x = torch.clamp(x, min=-clip_threshold, max=clip_threshold)
            
            local_std = x.std()
            x = x / std

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            dequantized_x = std * quantized_x
            
            updated_global_x = global_x + dequantized_x
            
        return updated_global_x, local_std


def WSQ_update(model, global_model, wt_bit, args):
    
    g_params = dict(global_model.named_parameters())
    
    global_std_values = {}
    for name, param in global_model.named_parameters():
        if "global_std" in name and name != "conv1.global_std":
            weight_name = name.replace("global_std", "weight")
            global_std_values[weight_name] = param.data  

    local_std_values = {}
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQConv2d(n_bits=wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = first_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQConv2d(n_bits=wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = layer_quant_conv(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQConv2d(n_bits=wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                updated_param, local_std = quant_conv1x1(param.data, g_params[name].data, global_std_values[name])
                param.data.copy_(updated_param)
                local_std_values[name] = local_std
                
    for name, param in model.named_parameters():
        if "local_std" in name:
            weight_name = name.replace("local_std", "weight")
            if weight_name in local_std_values:
                std = local_std_values[weight_name] 
                param.data.copy_(std)
