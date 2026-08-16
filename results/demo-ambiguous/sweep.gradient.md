# Crowded-neighbour sweep

Every row runs the identical opening plan: a grasp aimed 2.0 cm from the red
block's detected centre, inside `rule_verdict`'s 2.8 cm tolerance. Only the blue block's
distance changes between blocks of rows. `caught` counts scenes where the verifier rejected
the opening plan; `p (opening)` is the mean imagined success probability of that plan;
`red on pad` is what MuJoCo actually did.

| neighbour at | verifier | caught | p (opening) | red on pad | wrong block on pad | not executed |
| --- | --- | --- | --- | --- | --- | --- |
| 6.0 cm | none | 0/10 | — | 0/10 | 0/10 | 0/10 |
| 6.0 cm | rules | 0/10 | 1.000 | 0/10 | 0/10 | 0/10 |
| 6.0 cm | world-model | 1/10 | 0.822 | 0/10 | 0/10 | 1/10 |
| 7.5 cm | none | 0/10 | — | 0/10 | 0/10 | 0/10 |
| 7.5 cm | rules | 0/10 | 1.000 | 0/10 | 0/10 | 0/10 |
| 7.5 cm | world-model | 0/10 | 0.818 | 0/10 | 0/10 | 0/10 |
| 8.5 cm | none | 0/10 | — | 5/10 | 0/10 | 0/10 |
| 8.5 cm | rules | 0/10 | 1.000 | 5/10 | 0/10 | 0/10 |
| 8.5 cm | world-model | 0/10 | 0.837 | 5/10 | 0/10 | 0/10 |
| 22.0 cm | none | 0/10 | — | 10/10 | 0/10 | 0/10 |
| 22.0 cm | rules | 0/10 | 1.000 | 10/10 | 0/10 | 0/10 |
| 22.0 cm | world-model | 2/10 | 0.776 | 10/10 | 0/10 | 0/10 |
