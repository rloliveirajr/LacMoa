/*
 *    BasicClassificationSocoringEvaluator.java
 *    Copyright (C) 2013 Federal University of Minas Gerais, Belo Horizonte, Brazil
 *    @author Roberto L. Oliveira Junior (robertolojr at dcc dot ufmg dot br)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.evaluation;

import moa.AbstractMOAObject;
import moa.core.Measurement;
import weka.core.Instance;
import weka.core.Utils;

/**
 * Classification evaluator that performs basic incremental evaluation.
 * This class extends {@link BasicClassificationPerformanceEvaluator} including the MSE calculus.
 * 
 * @author Roberto L. Oliveira Junior (robertolojr at dcc dot ufmg dot br)
 * @version $Revision: 1 $
 */
public class BasicClassificationScoringEvaluator extends AbstractMOAObject
			implements ClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;
    
    protected double mse;    
    protected int saw;

    protected double weightObserved;

    protected double weightCorrect;

    protected double[] columnKappa;

    protected double[] rowKappa;

    protected int numClasses;

    @Override
    public void reset() {
        reset(this.numClasses);
    }

    public void reset(int numClasses) {
        this.numClasses = numClasses;
        this.rowKappa = new double[numClasses];
        this.columnKappa = new double[numClasses];
        for (int i = 0; i < this.numClasses; i++) {
            this.rowKappa[i] = 0.0;
            this.columnKappa[i] = 0.0;
        }
        this.weightObserved = 0.0;
        this.weightCorrect = 0.0;
    }

    @Override
    public void addResult(Instance inst, double[] classVotes) {
        double weight = inst.weight();
        int trueClass = (int) inst.classValue();
        if (weight > 0.0) {
            if (this.weightObserved == 0) {
                reset(inst.dataset().numClasses());
            }
            this.weightObserved += weight;
            int predictedClass = Utils.maxIndex(classVotes);
            if (predictedClass == trueClass) {
                this.weightCorrect += weight;
            }
            
            this.saw++;
            this.mse += (1-classVotes[trueClass])*(1-classVotes[trueClass]);
            this.rowKappa[predictedClass] += weight;
            this.columnKappa[trueClass] += weight;
        }
    }

    @Override
    public Measurement[] getPerformanceMeasurements() {
    	return new Measurement[]{
                new Measurement("classified instances",
                getTotalWeightObserved()),
                new Measurement("classifications correct (percent)",
                getFractionCorrectlyClassified() * 100.0),
                new Measurement("Kappa Statistic (percent)",
                getKappaStatistic() * 100.0),
                new Measurement("MSE (percent)",getMSE() * 100.0)};
    }
    
    public double getTotalWeightObserved() {
        return this.weightObserved;
    }

    public double getFractionCorrectlyClassified() {
        return this.weightObserved > 0.0 ? this.weightCorrect
                / this.weightObserved : 0.0;
    }

    public double getFractionIncorrectlyClassified() {
        return 1.0 - getFractionCorrectlyClassified();
    }

    public double getKappaStatistic() {
        if (this.weightObserved > 0.0) {
            double p0 = getFractionCorrectlyClassified();
            double pc = 0.0;
            for (int i = 0; i < this.numClasses; i++) {
                pc += (this.rowKappa[i] / this.weightObserved)
                        * (this.columnKappa[i] / this.weightObserved);
            }
            return (p0 - pc) / (1.0 - pc);
        } else {
            return 0;
        }
    }
    
    public double getMSE(){
		return (this.mse/(double)this.saw);
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        Measurement.getMeasurementsDescription(getPerformanceMeasurements(),
                sb, indent);
    }
}