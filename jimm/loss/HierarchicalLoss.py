import jittor as jt
import jittor.nn as nn
import jittor.nn as F
class HierarchicalLoss(nn.Module):
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss = base_loss_fn
        # 定义层级关系：0-1为良性，2-4为可疑恶性，5为阴性
        self.hierarchy = {
            'benign': [0, 1],
            'suspicious': [2, 3, 4],
            'negative': [5]
        }
    
    def forward(self, preds, targets):
        # 基础分类损失
        class_loss = self.base_loss(preds, targets)
        
        # 层级分类损失
        benign_probs = preds[:, self.hierarchy['benign']].sum(dim=1)
        suspicious_probs = preds[:, self.hierarchy['suspicious']].sum(dim=1)
        negative_probs = preds[:, self.hierarchy['negative']].squeeze()
        
        hierarchy_preds = jt.stack(
            [benign_probs, suspicious_probs, negative_probs], 
            dim=1
        )
        
        # 创建层级目标
        hierarchy_targets = jt.zeros_like(hierarchy_preds)
        for i, target in enumerate(targets):
            if target in self.hierarchy['benign']:
                hierarchy_targets[i, 0] = 1
            elif target in self.hierarchy['suspicious']:
                hierarchy_targets[i, 1] = 1
            else:
                hierarchy_targets[i, 2] = 1
        
        hierarchy_loss = F.binary_cross_entropy_with_logits(
            hierarchy_preds, hierarchy_targets
        )
        
        return class_loss + 0.3 * hierarchy_loss