# Demo sweep

The opening plan is the model's own — no flaw is scripted in — so this is a test of the planner's raw mistake rate, not just its ability to apply a rejection reason it was handed.
`caught` counts scenes where the verifier rejected the opening plan;
`success` counts scenes where the block finished on the pad in MuJoCo.

| scenario | verifier | caught | repaired to success | success rate |
| --- | --- | --- | --- | --- |
| grasp_miss | none | 0/3 | 3/3 | 1.00 |
| grasp_miss | world-model | 0/3 | 2/3 | 0.67 |
