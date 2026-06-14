#!/usr/bin/env python3
"""
Compat entrypoint for student knowledge distillation.

Usage mirrors the refactored Distillation_main.py:
python dl-student_main.py --config configs/Distill_model.yaml --model lstm --teacher-model tcn_transformer --teacher-checkpoint weights/best_tcn_transformer_binary_model.pth --epochs 10
"""

from Distillation_main import main


if __name__ == "__main__":
    main()
