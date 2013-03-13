package moa.classifiers.rules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import weka.core.Attribute;
import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.options.IntOption;

public class LACWPC extends AbstractClassifier{

private static final long serialVersionUID = 4740958383832856257L;

	
	public IntOption maxRuleSizeOption = new IntOption("maxRuleSize", 'r', "", 
			4, 0, Integer.MAX_VALUE);
	
	public boolean alwaysInsertOption = true;

	protected LACInstances trainingInstances;
	protected LACRules rules;
	private List<LACRule> previousRules;
	protected int window;
	
	Set<String> instancesRemoved;
	
	@Override
	public void resetLearningImpl() {
		trainingInstances = new LACInstances();
		previousRules = new ArrayList<LACRule>();
		alwaysInsertOption = true;
		window = 1;
	}

	@Override
	public void trainOnInstanceImpl(Instance wekaInstance) {
		boolean considerFeaturePositions = checkNominalAttributes(wekaInstance);
		
		instancesRemoved = new HashSet<String>();
		
		LACInstances newTraining = new LACInstances(considerFeaturePositions);
		
		Set<Integer> selected = new TreeSet<Integer>();
		Set<Integer> removed = new TreeSet<Integer>();
		
		int[] instances = new int[trainingInstances.length()];
		Arrays.fill(instances, 0);
		
		for(int r = 0; r < previousRules.size(); r++){
			LACRule rule = previousRules.get(r);
			List<Integer> ruleFeatures = rule.getFeaturesIndexed();
			int ruleClass = rule.getClassIndex();
			
			for(int i = 0; i < instances.length; i++){
				LACInstance inst = trainingInstances.getInstance(i);
				List<Integer> features = inst.getIndexedFeatures();
				int label = inst.getIndexedClass();
				
				if(label == ruleClass){
					boolean covered = true;
					for(int f = 0; f < ruleFeatures.size() && covered; f++){
						int ruleFeature = ruleFeatures.get(f);
						int pos = Collections.binarySearch(features, ruleFeature);
						
						if(pos < 0){
							covered = false;
						}
					}
					
					if(covered){
						instances[i] = instances[i] + 1;
					}
				}
			}
		}
		
		for(int i = 0; i < instances.length; i++){
			if(instances[i] > 0){
				selected.add(i);
			}else{
				removed.add(i);
			}
		}
		
//		for(int i = 0; i < trainingInstances.length(); i++){
//			LACInstance inst = trainingInstances.getInstance(i);
//			List<Integer> features = inst.getIndexedFeatures();
//			int clazz = inst.getIndexedClass();
//						
//			//Rules coverage
//			for(int r = 0; r < previousRules.size(); r++){
//			
//				LACRule rule = previousRules.get(r);
//				List<Integer> ruleFeatures = rule.getFeaturesIndexed();
//				int ruleClass = rule.getClassIndex();
//				
//				if(clazz == ruleClass){
//					boolean insert = true;
//					for(int f = 0; f < ruleFeatures.size(); f++){
//						int ruleFeature = ruleFeatures.get(f);
//						int pos = Collections.binarySearch(features, ruleFeature);
//						
//						if(pos < 0){
//							insert = false;
//							break;
//						}
//					}
//				}
//			}
//			
//			
//		}
		
		if(removed.size() > 0){
			for(Integer i : removed){
				LACInstance inst = trainingInstances.getInstance(i);
				instancesRemoved.add(inst.toString());
			}
		}
		
		
		if(selected.size() > 0){
			Set<String> classes = new HashSet<String>();
			for(Integer i : selected){
				LACInstance inst = newTraining.createNewTrainingInstance();

				List<Integer> featuresIndexed = trainingInstances.getInstance(i).getIndexedFeatures();
				String classLabel = trainingInstances.getInstance(i).getHiddenClazz().getLabel();
				
				for(Integer f : featuresIndexed){
					LACFeature feature = trainingInstances.getFeatureByIndex(f);
					inst.addFeature(feature.getLabel());
				}
				classes.add(classLabel);
				inst.setClass(classLabel);	
			}
						
			trainingInstances = null;
			trainingInstances = newTraining;
		}

		trainingInstances.considerFeaturePosition = considerFeaturePositions;
		
		if(alwaysInsertOption){
			LACInstance trainingInstance = trainingInstances.createNewTrainingInstance();
			populateInstance(wekaInstance, trainingInstance, true);
		}
		
		this.rules = this.trainingInstances.prepare(maxRuleSizeOption.getValue() - 1, considerFeaturePositions);
	}

	@Override
	public double[] getVotesForInstance(Instance wekaInstance) {

		double[] result = new double[wekaInstance.classAttribute().numValues()];
		
		if(trainingInstances.length() > 0){
			LACInstance testInstance = new LACInstance(trainingInstances);
			populateInstance(wekaInstance, testInstance, false);
			
			double[] probs = null;

			probs = calculateProbabilities(testInstance);
			
			for (int i = 0; i < probs.length; i++){
				String value = trainingInstances.getClassByIndex(i).getLabel();
				int index = wekaInstance.classAttribute().indexOfValue(value);
				if(index >= 0){
					result[index] = probs[i];
				}
			}
		}
			
		return result;
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
                new Measurement("(lac-wpc) window " + this.window,
                this.trainingInstances.length())};
	}
	
	/**
	 * Populates a {@link LACInstance} with the contents of an Weka {@link Instance}
	 * @param wekaInstance
	 * @param lacInstance
	 * @param populateClass 
	 */
	private void populateInstance(Instance wekaInstance, LACInstance lacInstance, boolean populateClass)
	{
		int numAtts = wekaInstance.numAttributes();
		for (int i = 0; i < numAtts; i++)
		{
			if(i != wekaInstance.classIndex())
			{
				String label = wekaInstance.toString(i);
				lacInstance.addFeature(label);
			}
			else 
			{
				if(populateClass)
				{					
					String clazz = wekaInstance.classAttribute().value((int) wekaInstance.classValue());
					lacInstance.setClass(clazz);
				}
				else
				{
					String clazz = wekaInstance.classAttribute().value((int) wekaInstance.classValue());
					lacInstance.setHiddenClass(clazz);
				}
			}
		}	
	}
	
	/**
	 * Returns true if all attributes are nominal or false
	 * if all of them are string. Throws a runtime exception
	 * for mixed attribute types.
	 * 
	 * @param data
	 * @return
	 */
	private boolean checkNominalAttributes(Instance data)
	{
		boolean hasNominalAtt = false;
		boolean hasStringAtt = false;
		
		for(int i = 0; i < data.numAttributes(); i++)
		{
			if(data.classIndex() != i)
			{
				Attribute att = data.attribute(i);
				hasNominalAtt = hasNominalAtt || att.isNominal();
				hasStringAtt = hasStringAtt || att.isString();
			}
		}
		
		if(hasNominalAtt && hasStringAtt)
		{
			throw new RuntimeException("Lazy Associative Classifiers can only handle datasets were all attributes have the same type. Make sure all attributes are either string or nominal.");
		}
		
		return hasNominalAtt;
	}
	
	/**
	 * Gets the probability of the given test instance belonging to each class.
	 * 
	 * @param testInstance
	 * @throws Exception 
	 */
	double[] calculateProbabilities(LACInstance testInstance) {
		double[] probs;
		double[] scores = calculateScores(testInstance);
		
		if(scores != null) {
			probs = new double[scores.length];
			double scoreSum = 0.0;
			for (int i = 0; i < scores.length; i++) {
				scoreSum += scores[i];
			}
			
			for (int i = 0; i < scores.length; i++) {
				probs[i] = scores[i] / scoreSum;
			}
		}else {
			Set<Integer> allClasses = trainingInstances.getAllClasses();
			probs = new double[allClasses.size()];
			for (Integer clazz : allClasses) {
				double count = trainingInstances.getInstancesOfClass(clazz).size();
				probs[clazz] = (count / ((double) trainingInstances.length()));
			}
		}

		return probs ;
	}

	
	private double[] calculateScores(LACInstance testInstance) {
		
		List<LACRule> allRulesForFeatures = rules.extractAllRules(testInstance);
		
		int numClasses = trainingInstances.getAllClasses().size();
		double[] scores = new double[numClasses];
		int numRules = allRulesForFeatures.size();
		
		Skyline sb = new Skyline();
		
		for(int i = 0; i < numRules; i++){
			sb.addRule(allRulesForFeatures.get(i));
		}

		previousRules = sb.window;
		
		if(allRulesForFeatures.size() > 0) {
						
			for(LACRule w : allRulesForFeatures) {
				scores[w.getPredictedClass()] = scores[w.getPredictedClass()] + w.getConfidence();
			}
			
		}else {
			scores = null;
		}
	
		return scores;
	}
}
