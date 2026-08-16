# Crowded-neighbour sweep

Every row runs the identical opening plan: a grasp aimed 2.7 cm from the red
block's detected centre, inside `rule_verdict`'s 2.8 cm tolerance. Only the blue block's
distance changes between blocks of rows. `caught` counts scenes where the verifier rejected
the opening plan; `p (opening)` is the mean imagined success probability of that plan;
`red on pad` is what MuJoCo actually did.

| neighbour at | verifier | caught | p (opening) | red on pad | wrong block on pad | not executed |
| --- | --- | --- | --- | --- | --- | --- |
| 4.2 cm | none | 0/12 | — | 0/12 | 9/12 | 0/12 |
| 4.2 cm | rules | 0/12 | 1.000 | 0/12 | 9/12 | 0/12 |
| 4.2 cm | world-model | 0/12 | 0.795 | 0/12 | 9/12 | 0/12 |
