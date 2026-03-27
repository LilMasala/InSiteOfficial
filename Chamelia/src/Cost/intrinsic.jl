""" 
intrinsic.jl
Intrinsic cost module -normative cost C^int_t 
Hardwired and audible. No epistemic term 

Answers: How bad is this state of affairs for this patient 

There are four components:
C^int = C^physical + C^burden + C^burnout + C^trust

C^physical is domain-specific — registered by simulator plugin.


""" 



# The explicit intrinsic-cost helpers are owned by the shared type layer in
# `src/types.jl` and imported into `Cost`. This file remains as the module-local
# documentation anchor for that interface.
