from loguru import logger

from arguments import OptimizationParams

from ortho_gaussian_renderer import GenerateMode


class TrainingController:
    def __init__(self, opt_params: OptimizationParams):
        self.current_iteration = 0
        self.opt_params = opt_params
        self._entropy_constrained = False

        self._previous_render_mode = None
        self._prev_gaussian_statis = False
        self._prev_gaussian_adjust_anchor = False

    @property
    def render_mode(self):

        if self.current_iteration <= self.opt_params.full_precision_training_total:
            new_mode = GenerateMode.TRAINING_FULL_PRECISION
        elif self.current_iteration <= self.opt_params.full_precision_training_total + self.opt_params.quantized_training_total:
            new_mode = GenerateMode.TRAINING_QUANTIZED
        elif self.current_iteration <= self.opt_params.full_precision_training_total \
                + self.opt_params.quantized_training_total \
                + self.opt_params.entropy_constrained_train_total:
            self._entropy_constrained = True
            new_mode = GenerateMode.TRAINING_ENTROPY

        elif self.current_iteration <= self.opt_params.full_precision_training_total \
                + self.opt_params.quantized_training_total \
                + self.opt_params.entropy_constrained_train_total \
                + self.opt_params.ste_entropy_constrained_train_total:
            self._entropy_constrained = True
            new_mode = GenerateMode.TRAININ_STE_ENTROPY
        else:
            new_mode = None

        if new_mode != self._previous_render_mode:
            logger.info(f'switch render mode at {self.current_iteration}: {self._previous_render_mode} -> {new_mode}')
            self._previous_render_mode = new_mode

        return new_mode

    @property
    def entropy_constrained(self):
        return self._entropy_constrained

    @property
    def gaussian_statis(self):
        # 在刚开始量化时不记录梯度信息
        if self.opt_params.full_precision_training_total <= self.current_iteration < self.opt_params.full_precision_training_total + self.opt_params.pause_densification:
            new_mode = False
        elif self.opt_params.update_until > self.current_iteration > self.opt_params.start_stat:
            new_mode = True
        else:
            new_mode = False

        if new_mode != self._prev_gaussian_statis:
            logger.info(f'switch gaussian_statis at {self.current_iteration}: {self._prev_gaussian_statis} -> {new_mode}')
            self._prev_gaussian_statis = new_mode

        return new_mode

    @property
    def gaussian_adjust_anchor(self):
        if self.current_iteration >= self.opt_params.update_until:
            new_mode = False

        elif self.opt_params.full_precision_training_total <= self.current_iteration <= self.opt_params.full_precision_training_total + self.opt_params.pause_densification:
            # 在开始量化后的1000个iteration内不执行anchor调整
            new_mode = False

        elif self.current_iteration > self.opt_params.update_from and self.current_iteration % self.opt_params.update_interval == 0:
            new_mode = True

        else:
            new_mode = False

        if new_mode != self._prev_gaussian_adjust_anchor:
            logger.info(
                f'switch gaussian_adjust_anchor at {self.current_iteration}: {self._prev_gaussian_adjust_anchor} -> {new_mode}')
            self._prev_gaussian_adjust_anchor = new_mode
        return new_mode

    @property
    def clean_denorm(self):
        return self.current_iteration == self.opt_params.update_until

    def step(self):
        self.current_iteration += 1
