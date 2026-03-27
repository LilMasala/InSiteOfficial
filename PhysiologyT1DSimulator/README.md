# PhysiologyT1DSimulator

## MVP Chamelia simulation runner

Run the minimum honest pre-JEPA Chamelia simulation with:

```bash
python -m chamelia.run_simulation \
  --n-patients 200 \
  --days 180 \
  --seed 42 \
  --outdb artifacts/chamelia_sim.db \
  --report artifacts/chamelia_report.json
```

This canonical path uses:

- synthetic patients from `t1d_sim.population`
- the existing `chamelia.run.WorldRunner` daily orchestration
- the aggregate predictor + grid-search optimizer only
- observation → shadow → intervention lifecycle phases
- simulated recommendation acceptance/rejection
- persistent SQLite outputs plus a JSON summary report

The JSON report explicitly answers whether mean TIR improved, whether burnout
remained acceptable, how burnout was defined, and how many patients entered
shadow or intervention.
