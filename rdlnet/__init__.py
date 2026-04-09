"""RDLNet: Real-world Document Localization Network (ACM MM 2024)."""

from .distill import (
    DistillConfig,
    LightSAMMultiplexDistillation,
    build_teacher_image_encoder_vit_h,
    create_distillation_setup,
    distill_trainable_state_dict,
    load_distill_trainable_state_dict,
    load_distilled_student_into_rdlnet,
    load_student_encoder_into_rdlnet_from_checkpoint,
    load_teacher_weights_from_sam_checkpoint,
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
    "distill_trainable_state_dict",
    "load_distill_trainable_state_dict",
    "load_student_encoder_into_rdlnet_from_checkpoint",
]
