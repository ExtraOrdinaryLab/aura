import os
import shutil

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from ..logger import console


class SavePeftModelCallback(TrainerCallback):
    """
    Callback function to save the best-performing model during training
    """
    def on_save(
        self,
        training_args: TrainingArguments,
        trainer_state: TrainerState,
        trainer_control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """
        This method is triggered when saving a checkpoint during training.
        It ensures that the best-performing model is saved in a dedicated folder.

        Args:
            training_args (TrainingArguments): The training arguments including output directory and rank.
            trainer_state (TrainerState): The current state of the trainer, including the best model checkpoint.
            trainer_control (TrainerControl): The control object to manage the trainer's behavior.
        Returns:
            TrainerControl: The control object to manage the trainer's behavior.
        """
        # Ensure that the operation is performed only on the main process
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            # Define the folder to store the best-performing model
            best_checkpoint_folder = os.path.join(
                training_args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best"
            )

            # Check if there is a best model checkpoint and it exists on disk
            if (
                trainer_state.best_model_checkpoint is not None
                and os.path.exists(trainer_state.best_model_checkpoint)
            ):
                # Remove the previous best checkpoint folder if it exists
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)

                # Copy the best model checkpoint to the best checkpoint folder
                shutil.copytree(trainer_state.best_model_checkpoint, best_checkpoint_folder)

            # Log the best model checkpoint and its evaluation metric
            console.log(
                f"The best checkpoint is located at: {trainer_state.best_model_checkpoint}, "
                f"with evaluation metric: {trainer_state.best_metric}"
            )

        return trainer_control
