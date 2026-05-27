import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        """
        Custom loss function combining hard label Cross-Entropy and soft target KL-Divergence.
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # KL Divergence reduction is set to batchmean as required by PyTorch standard
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): Output logit distribution from the Student model.
            teacher_logits (torch.Tensor): Output logit distribution from the Teacher model.
            y_true (torch.Tensor): True hard labels (0, 1, 2).
        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        # 1. Calculate standard hard label classification loss
        hard_loss = self.cross_entropy(student_logits, y_true)
        
        # 2. Calculate soft target loss using KL-Divergence with Temperature scaling
        # Softmax with temperature for the Teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        # Log-Softmax with temperature for the Student (required input for PyTorch KLDivLoss)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Scale the soft loss component by T^2 to maintain gradient scale consistency
        soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 3. Linearly combine both losses based on the alpha weight factor
        combined_loss = (self.alpha * soft_loss) + ((1.0 - self.alpha) * hard_loss)
        
        return combined_loss