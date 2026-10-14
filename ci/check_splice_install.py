"""Install the pinned splice model for the synthetic-reference inference check."""
from aso_agents.model_sources import install_source
from aso_agents.splice_eval_assets import (splicetransformer_root,
    SPLICETRANSFORMER_REVISION, _ensure_checkpoint)

root = splicetransformer_root()
install_source("https://github.com/ShenLab-Genomics/SpliceTransformer.git", SPLICETRANSFORMER_REVISION, root)
_ensure_checkpoint(root, print)

from aso_agents.openspliceai_assets import install_mouse_model
install_mouse_model()
