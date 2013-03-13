package moa.classifiers.rules;

import java.util.Arrays;
import java.util.List;
import java.util.Stack;

public class Skyline {

	private enum RELATION {DOMINATED, DOMINATES, INCOMPARABLE};
	
	protected Stack<LACRule> window;
	
	public Skyline(){
		window = new Stack<LACRule>();
	}
	
	public void addRule(LACRule point){
		int dominated = 1;
		int dominates = 0;
		int incomparable = 2;
		
		int[] dominance = new int[window.size()];
		Arrays.fill(dominance, 0);
		
		for(int i = 0; i < dominance.length; i++){
			LACRule w = window.get(i);
			
			RELATION r = compare(point, w);
			
			if(r == RELATION.DOMINATED){
				dominance[i] = dominates;
			}else if(r == RELATION.DOMINATES){
				dominance[i] = dominated;
			}else{
				dominance[i] = incomparable;
			}
		}
		
		boolean pointDominates = false;
		int allIncomparable = 0;

		//Quando uma regra eh dominada ela eh removida
		for(int j = dominance.length - 1; j > -1; j--){
			if(dominance[j] == dominated){
				window.remove(j);
				pointDominates = true;
			}
			
			if(dominance[j] == incomparable){
				allIncomparable++;
			}
		}
		
		if(pointDominates || window.isEmpty() || allIncomparable == window.size()){
			window.push(point);
		}		
	}
	
	private static Skyline.RELATION compare(LACRule point1, LACRule point2){
		int i = 0;
		
		double[] metricsPoint1 = point1.getMetrics();
		double[] metricsPoint2 = point2.getMetrics();
		
		while(i < metricsPoint1.length && cmp(metricsPoint1[i], metricsPoint2[i]) == 0){
			i++;
		}

		if(i == metricsPoint1.length){
			return RELATION.INCOMPARABLE;
		}
		
		if(cmp(metricsPoint1[i], metricsPoint2[i]) < 0){
			for(++i; i < metricsPoint1.length; i++){
				if(cmp(metricsPoint1[i], metricsPoint2[i]) > 0){
					return RELATION.INCOMPARABLE;
				}
			}
			return RELATION.DOMINATED;
		}
			
		for(++i; i < metricsPoint1.length; i++){
			if(cmp(metricsPoint1[i], metricsPoint2[i]) > 0){
				return RELATION.INCOMPARABLE;
			}
		}
		return RELATION.DOMINATES;
	}
	
	private static int cmp(double a, double b){
	    double diff = a - b;

	    if(diff < 0){
	    	return -1;
	    }else if(diff > 0){
	    	return 1;
	    }
	    
	    return 0;
    }
	
	public List<LACRule> getWindow(){
		return this.window;
	}
}
