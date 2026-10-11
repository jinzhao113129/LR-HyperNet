"""修复 local_remote_grouper.py：添加 EuclideanRemoteAggregator + 分支逻辑"""

filepath = 'models/local_remote_grouper.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 在 HyperbolicRemoteAggregator 类后面、LocalRemoteGrouper 前面插入 EuclideanRemoteAggregator
euclidean_class = '''

class EuclideanRemoteAggregator(nn.Module):
    """远程邻域欧氏聚合（双曲版的等价对照，消融实验用）。

    与 HyperbolicRemoteAggregator 逐项同构：同 anchor（组内欧氏质心）、
    同变换-注意力加权聚合-out_proj+残差+LayerNorm 结构，
    唯一区别：距离用欧氏平方距离替代双曲距离，
    温度为可学习标量（替代可学习曲率）。参数量与双曲版完全一致。

    输入: [B, S, G, C]（G 个远程事件的特征）
    输出: [B, S, C]（欧氏质心聚合后的特征）
    """

    def __init__(self, hidden_size: int, temperature_init: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.mobius_linear = nn.Linear(hidden_size, hidden_size)
        self.mobius_bias = nn.Parameter(torch.zeros(hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _temperature(self) -> torch.Tensor:
        """正温度 T = abs(T_param) + 1e-4"""
        return torch.abs(self.temperature) + 1e-4

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """单个组的欧氏聚合。

        Args:
            x: [M, K, C]，K 个点的欧氏特征
        Returns:
            [M, C]，聚合后的欧氏特征
        """
        T = self._temperature()
        centroid = x.mean(dim=1, keepdim=True)
        distances = ((x - centroid) ** 2).sum(dim=-1)
        attn_weights = F.softmax(-distances / T, dim=-1)
        x_transformed = self.mobius_linear(x) + self.mobius_bias
        agg = torch.bmm(attn_weights.unsqueeze(1), x_transformed).squeeze(1)
        out = self.out_proj(agg)
        residual = x.mean(dim=1)
        return self.layer_norm(out + residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, G, C = x.shape
        out = self._aggregate(x.reshape(-1, G, C))
        return out.reshape(B, S, C)

'''

insert_point = '# ============================================================================\n# Local+Remote 分组器'
content = content.replace(insert_point, euclidean_class + '\n' + insert_point)

# 2. 修改 LocalRemoteGrouper.__init__ 中的 hyper_agg 创建，加分支
old_agg = """        self.hyper_agg = HyperbolicRemoteAggregator(channel, curvature_init=curvature_init,
                                                     hyperbolic_type=hyperbolic_type)"""
new_agg = """        if hyperbolic_type == "euclidean":
            self.hyper_agg = EuclideanRemoteAggregator(channel, temperature_init=1.0)
        else:
            self.hyper_agg = HyperbolicRemoteAggregator(channel, curvature_init=curvature_init,
                                                         hyperbolic_type=hyperbolic_type)"""
content = content.replace(old_agg, new_agg)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed: EuclideanRemoteAggregator added and LocalRemoteGrouper branched")
