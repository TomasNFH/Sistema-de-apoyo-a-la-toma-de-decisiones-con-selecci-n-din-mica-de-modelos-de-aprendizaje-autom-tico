from System import Workflow
import warnings
import cutie

Fast = cutie.prompt_yes_or_no("Fast or Slow mode?")
if not Fast: warnings.filterwarnings("ignore")

ret = Workflow.dyn_model_selection(FAST = Fast, PLOT = True, local_file=False)
breakpoint()
