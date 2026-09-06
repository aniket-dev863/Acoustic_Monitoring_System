/**
 * Experiment Results Controller.
 * Exposes benchmark results (gate comparison and 6-stage ablations).
 */

const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '../../../');
const RESULTS_DIR = path.join(ROOT_DIR, 'experiments/results');

exports.getExperiments = (req, res, next) => {
  try {
    const comparisonPath = path.join(RESULTS_DIR, 'gate_comparison.json');
    const ablationPath = path.join(RESULTS_DIR, 'ablation_study.json');

    let gateComparison = null;
    let ablationStudy = null;

    if (fs.existsSync(comparisonPath)) {
      gateComparison = JSON.parse(fs.readFileSync(comparisonPath, 'utf8'));
    }

    if (fs.existsSync(ablationPath)) {
      ablationStudy = JSON.parse(fs.readFileSync(ablationPath, 'utf8'));
    }

    res.json({
      gate_comparison: gateComparison,
      ablation_study: ablationStudy,
    });
  } catch (err) {
    next(err);
  }
};
