# Demo sweep

The same two flawed opening plans over 8 scenes, all three arms, identical scenes across arms.
`caught` counts scenes where the verifier rejected the flawed opening plan;
`success` counts scenes where the block finished on the pad in MuJoCo.

| scenario | verifier | caught | repaired to success | success rate |
| --- | --- | --- | --- | --- |
| grasp_miss | none | 0/8 | 0/8 | 0.00 |
| grasp_miss | rules | 8/8 | 7/8 | 0.88 |
| grasp_miss | world-model | 8/8 | 6/8 | 0.75 |
| place_miss | none | 0/8 | 0/8 | 0.00 |
| place_miss | rules | 8/8 | 7/8 | 0.88 |
| place_miss | world-model | 8/8 | 6/8 | 0.75 |
