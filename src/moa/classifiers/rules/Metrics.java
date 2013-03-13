package moa.classifiers.rules;

import java.util.Arrays;

public class Metrics {

	private static final double EPS = 1e-15;
	
	double omega;
	double antecedentSupport;
	double classSupport;
	double ruleSupport;
	
	double[] classesSupport;
	
	double anb,nanb,nab;
	
	Metrics(double[] classesSupport){
		this.classesSupport = classesSupport;
		
		omega = 0;
		
		for(int i = 0; i < classesSupport.length; i++){
			omega += classesSupport[i];
		}
	}
	
	void setAntecedentSupport(double pa){
		this.antecedentSupport = pa + EPS;
	}
	
	void setRuleSupport(double pab, int classID){
		this.ruleSupport = pab;
		this.classSupport = this.classesSupport[classID];
		
		this.anb = this.antecedentSupport - this.ruleSupport;
		this.nanb = this.omega - this.classSupport - anb;
	}
	
	double confidence(){
		return this.ruleSupport/this.antecedentSupport;
	}

	double support(){
		 return this.ruleSupport/this.omega;
	}

	double addedValue() {
		return (confidence() - (this.classSupport/omega));
	}

	double certainty(){
		return (this.ruleSupport/this.antecedentSupport - this.classSupport/omega)/(EPS + 1. - this.classSupport/omega);
	}

	double yulesQ(){
		return  (this.ruleSupport*nanb - anb*nab)/(EPS + this.ruleSupport*nanb + anb*nab);
	}

	double yulesY(){
		return (Math.sqrt(this.ruleSupport*nanb) - Math.sqrt(anb*nab))/(EPS + Math.sqrt(this.ruleSupport*nanb) + Math.sqrt(anb*nab));
	}

	double strengthScore(){
		double nb = omega - this.classSupport;
		return this.ruleSupport/(this.classSupport + EPS) * confidence() / (EPS + anb/nb);
	}

	double weightedRelativeConfidence() {
		return (confidence() - this.classSupport/omega)*this.antecedentSupport/omega;
	}
	
	double[] getMetrics(){
		double[] metrics = new double[7];
		metrics[0] = confidence();
		metrics[1] = support();
//		metrics[2] = addedValue();
		metrics[2] = certainty();
		metrics[3] = yulesQ();
		metrics[4] = yulesY();
		metrics[5] = strengthScore();
		metrics[6] = weightedRelativeConfidence();
		
		return metrics;
	}
	
	public String toString(){
		return Arrays.toString(getMetrics());
	}
}
