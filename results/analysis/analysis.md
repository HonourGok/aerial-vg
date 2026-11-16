## Pr@0.5 by ground-truth box area (three equal groups by rank)

Group edges (share of image area): small < 0.52%, medium < 4.88%

| group | n | baseline | d4_remap | no_geo | d4_noremap |
|---|---|---|---|---|---|
| small | 2500 | 2.6 | 4.5 | 3.5 | 0.1 |
| medium | 2500 | 52.7 | 54.7 | 51.6 | 21.9 |
| large | 2500 | 84.0 | 85.8 | 81.8 | 62.6 |

## Share of small-group expressions that mention a class word (substring match)

vehicle 49.2%, windmill 11.6%, bridge 11.2%, ship 5.0%, storage tank 1.8%, airplane 1.6%, chimney 1.1%

## Centre of the box lies in the half named by the expression (%)

| subset | n | ground truth | baseline | d4_remap | no_geo | d4_noremap |
|---|---|---|---|---|---|---|
| left only | 1506 | 92.4 | 94.6 | 93.5 | 95.8 | 62.2 |
| right only | 1555 | 92.3 | 94.2 | 95.1 | 94.2 | 63.5 |
| top/upper only | 1396 | 91.5 | 93.2 | 92.2 | 92.9 | 70.8 |
| bottom/lower only | 1414 | 92.4 | 93.1 | 92.8 | 93.3 | 55.2 |

## d4_remap vs baseline at IoU 0.5

{"prop_only_correct": 445, "ref_only_correct": 303, "mcnemar_p": 2.3365077513457793e-07, "ref_correct_prop_below_0.3": 82, "prop_correct_ref_below_0.3": 115}

## Validation Pr@0.5 per epoch (first 500 validation queries)

- baseline: 12.0 → 23.4 → 27.6 → 30.6 → 35.2 → 41.2 → 42.0
- d4_remap: 15.2 → 21.6 → 27.6 → 32.0 → 36.8 → 43.6 → 44.2
- no_geo: 17.6 → 25.8 → 30.6 → 32.4 → 33.2 → 42.0 → 45.0
- d4_noremap: 13.2 → 18.0 → 16.4 → 23.2 → 22.2 → 26.8 → 26.2
