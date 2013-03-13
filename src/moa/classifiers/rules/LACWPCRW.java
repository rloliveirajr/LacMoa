package moa.classifiers.rules;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.IntOption;

public class LACWPCRW extends AbstractClassifier{

private static final long serialVersionUID = 4740958383832856257L;

	
	public IntOption maxRuleSizeOption = new IntOption("maxRuleSize", 'R', "", 
			4, 0, Integer.MAX_VALUE);

	private LACWPC window1;
	private LACWPC window2;
	private Random rand;
	private Set<String> dropped;
	
	@Override
	public void resetLearningImpl() {
		window1 = new LACWPC();
		window1.resetLearning();
		window1.maxRuleSizeOption = maxRuleSizeOption;
		window1.window = 1;
		
		window2 = new LACWPC();
		window2.resetLearning();
		window2.maxRuleSizeOption = maxRuleSizeOption;
		window2.alwaysInsertOption = false;
		window2.window = 2;
		
		rand = new Random(this.randomSeedOption.getValue());
	}

	@Override
	public void trainOnInstanceImpl(Instance wekaInstance) {
		window1.trainOnInstance(wekaInstance);
		dropped = window1.instancesRemoved;

		window2.trainOnInstance(wekaInstance);

		LACInstances window2Train = window2.trainingInstances;
		
		LACInstances newWindow2Train = new LACInstances(window2Train.considerFeaturePosition);

		int initialSize = window2Train.length();
		int finalSize = initialSize + dropped.size();
		
		int sizeWindow2 = sizeRandomWindow(initialSize, finalSize);
		
		for(int i = 0; i < window2Train.length(); i++){
			dropped.add(window2Train.getInstance(i).toString());
		}
		
		List<String> instancesWindow2 = new ArrayList<String>();
		instancesWindow2.addAll(dropped);
		
		Collections.shuffle(instancesWindow2, rand);
		
		for(int j = 0; j < instancesWindow2.size() && j < sizeWindow2; j++){
			String d = instancesWindow2.get(j);

			String[] features = d.split(",");

			String label = features[features.length - 1];
			LACInstance instance = newWindow2Train.createNewTrainingInstance();
			for(int i = 0; i < features.length - 1; i++){
				String f = features[i];
				instance.addFeature(f);				
			}
			instance.setClass(label);
		}

		window2.trainingInstances = newWindow2Train;
		window2.rules = window2.trainingInstances.prepare(maxRuleSizeOption.getMinValue(), window2Train.considerFeaturePosition);
	}

	@Override
	public double[] getVotesForInstance(Instance wekaInstance) {
		double[] probsWindow1 = window1.getVotesForInstance(wekaInstance);
		double[] probsWindow2 = window2.getVotesForInstance(wekaInstance);
		
		double[] probs = new double[wekaInstance.numClasses()];
		
		for(int i = 0; i < wekaInstance.numClasses(); i++){
			probs[i] = ((probsWindow1[i] + probsWindow2[i])) / 2;
		}
				
		return probs;
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{
				new Measurement("window 1", window1.trainingInstances.length()),
				new Measurement("window 2", window2.trainingInstances.length())};
	}
	
	private int sizeRandomWindow(int initialSize, int finalSize){
		
		double prop = (int)(Math.random() * (finalSize - initialSize)) + initialSize;

		int newSize = 0; 
		if(finalSize == 0){
			newSize = 0;
		} else {
			double size =  (prop);
			newSize = (int)(size);			
		}

		return newSize;
	}
}
