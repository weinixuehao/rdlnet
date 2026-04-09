"""RDLNet: Real-world Document Localization Network (ACM MM 2024)."""

from .distill import (
    DistillConfig,
    LightSAMMultiplexDistillation,
    create_distillation_setup,
    load_distilled_student_into_rdlnet,
    load_teacher_weights_from_sam_checkpoint,
    build_teacher_image_encoder_vit_h,
)
from .losses import RDLNetLoss, build_matcher
from .model import RDLNet, RDLNetConfig

__all__ = [
    "RDLNet",
    "RDLNetConfig",
    "RDLNetLoss",
    "build_matcher",
    "DistillConfig",
    "LightSAMMultiplexDistillation",
    "create_distillation_setup",
    "load_teacher_weights_from_sam_checkpoint",
    "build_teacher_image_encoder_vit_h",
    "load_distilled_student_into_rdlnet",
]
