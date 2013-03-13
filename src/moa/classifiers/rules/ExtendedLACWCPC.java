package moa.classifiers.rules;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.IntOption;

public class ExtendedLACWCPC extends AbstractClassifier{

private static final long serialVersionUID = 4740958383832856257L;

	
	public IntOption maxRuleSizeOption = new IntOption("maxRuleSize", 'r', "", 
			4, 0, Integer.MAX_VALUE);

	private LACWPC window1;
	private LACWPC window2;
	
	@Override
	public void resetLearningImpl() {
		window1 = new LACWPC();
		window1.resetLearning();
		window1.maxRuleSizeOption = maxRuleSizeOption;
		
		window2 = new LACWPC();
		window2.resetLearning();
		window2.maxRuleSizeOption = maxRuleSizeOption;
		window2.alwaysInsertOption = false;
	}

	@Override
	public void trainOnInstanceImpl(Instance wekaInstance) {
		window1.trainOnInstance(wekaInstance);
		LACInstances window1Train = window1.trainingInstances;
		
		Set<String> dropped = window1.instancesRemoved;

		window2.trainOnInstance(wekaInstance);

		LACInstances window2Train = window2.trainingInstances;
		
		for(int i = 0; i < window2Train.length(); i++){
			LACInstance inst = window1Train.createNewTrainingInstance();
			
			LACInstance old = window2Train.getInstance(i);
			List<Integer> indexedFeatures = old.getIndexedFeatures();
			
			for(int j : indexedFeatures){
				String label = window2Train.getFeatureByIndex(j).getLabel();
				
				inst.addFeature(label);
			}
			
			inst.setClass(old.getClazz().getLabel());
		}
		
		List<String> instancesWindow2 = new ArrayList<String>();
		instancesWindow2.addAll(dropped);
				
		LACInstances newWindow2Train = new LACInstances(window2Train.considerFeaturePosition);
		
		for(int j = 0; j < instancesWindow2.size(); j++){
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
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[]{
				new Measurement("(lac-wpc) window 1", window1.trainingInstances.length()),
				new Measurement("(lac-wpc) window 2", window2.trainingInstances.length())};		
	}
}
