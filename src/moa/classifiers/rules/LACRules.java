/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */
package moa.classifiers.rules;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Extracts classification rules from training data.
 * 
 * @author Gesse Dafe (Java implementation)
 * @author Adriano Veloso (algorithm and original C++ implementation)
 */
public class LACRules implements Serializable
{
	private static final long serialVersionUID = 5153978224002423432L;
	
	private final int maxRuleSize;
	private LACInstances trainingSet;
	
	/**
	 * Restricted access constructor
	 * @param training
	 * @param maxRuleSize
	 * @param minSupport
	 * @param minConfidence
	 * @param debug 
	 * @param outFile 
	 * @throws Exception 
	 */
	LACRules(LACInstances training, int maxRuleSize)
	{
		this.trainingSet = training;
		this.maxRuleSize = maxRuleSize;
	}

	/**
	 * Calculates the scores for each class instance.
	 * 
	 * @param testInstance
	 * @param currentClass
	 * @return
	 * @throws Exception 
	 */
	List<LACRule> extractAllRules(LACInstance testInstance)
	{
		List<Integer> testInstanceFeatures = new ArrayList<Integer>();
		testInstanceFeatures.addAll(testInstance.getIndexedFeatures());
		Collections.sort(testInstanceFeatures);
		
		List<LACRule> allRulesForFeatures = new ArrayList<LACRule>(10000);
		int[] numPatterns = {0};
		for(int i = 0; i < testInstanceFeatures.size(); i++)
		{
			List<Integer> featCombination = new ArrayList<Integer>();
			featCombination.add(testInstanceFeatures.get(i));
			extractRules(featCombination, testInstanceFeatures, allRulesForFeatures, numPatterns);
		}
		
		return allRulesForFeatures;
		
	}
	
	/**
	 * Recursively generates all subsets of an array
	 * @param numPatterns 
	 * 
	 * @param set
	 * @param maxSubsetSize
	 * @param subset
	 * @param result
	 * @return
	 */
	private void extractRules(List<Integer> pattern, List<Integer> testFeatures, List<LACRule> extractedRules, int[] numPatterns)
	{
		numPatterns[0]++;
		List<LACRule> rules = getRules(pattern);
			
		if(rules != null && rules.size() > 0)
		{
			extractedRules.addAll(rules);
			
			if(pattern.size() < maxRuleSize)
			{
				List<List<Integer>> combinations = new ArrayList<List<Integer>>();
				int size = testFeatures.size();
				for (int i = 0; i < size; i++)
				{
					int element = testFeatures.get(i);
					if (mustAddElement(element, pattern))
					{
						List<Integer> newFeatCombination = new ArrayList<Integer>(pattern.size() + 1);
						newFeatCombination.addAll(pattern);
						newFeatCombination.add(element);
						combinations.add(newFeatCombination);
					}
				}
				
				int numCombinations = combinations.size();
				for(int i = 0; i < numCombinations; i++)
				{
					extractRules(combinations.get(i), testFeatures, extractedRules, numPatterns);
				}
			}
		}
	}
	
	private List<LACRule> getRules(List<Integer> featuresCombination)
	{
		List<LACRule> rulesForFeatures = doExtractRules(featuresCombination);
		
		return rulesForFeatures;
	}

	/**
	 * Returns true is an element must be stored in the list
	 * 
	 * @param element
	 * @param featuresCombination
	 */
	private boolean mustAddElement(int element, List<Integer> featuresCombination)
	{
		return featuresCombination.size() < maxRuleSize && featuresCombination.get(featuresCombination.size() - 1) < element;
	}

	/**
	 * Extracts the applicable rules for the given combination of features.
	 * 
	 * @param featuresCombination
	 */
	private List<LACRule> doExtractRules(List<Integer> featuresCombination)
	{
		List<LACRule> rules = new ArrayList<LACRule>();

		List<Integer> instancesWithFeatures = trainingSet.getInstancesWithFeatures(featuresCombination);
		int numClasses = trainingSet.getAllClasses().size();
		int[] count = new int[numClasses];

		int size = instancesWithFeatures.size();

		if (size > 0)
		{

			for (int i = 0; i < size; i++)
			{
				Integer instanceIndex = instancesWithFeatures.get(i);
				LACInstance instance = trainingSet.getInstance(instanceIndex);
				int predictedClass = instance.getIndexedClass();
				count[predictedClass] = count[predictedClass] + 1;
			}	
			
			double[] classesSupport = new double[count.length];
			
			for(int i = 0; i < numClasses; i++){
				classesSupport[i] = count[i]/(double)size;
			}
			
			for (int i = 0; i < numClasses; i++) {
				
				Metrics metrics = new Metrics(classesSupport);
				
				double ab = classesSupport[i];
				double pa = (double) instancesWithFeatures.size() / (double)trainingSet.length();
				
				metrics.setRuleSupport(ab, i);
				metrics.setAntecedentSupport(pa);
				
				LACRule rule = new LACRule(metrics, i);
				rule.setPattern(trainingSet.indexesToLabels(featuresCombination));
				rule.setFeaturesIndexed(featuresCombination);
				rule.setClassLabel(trainingSet.getClassByIndex(i).getLabel());
				
				rules.add(rule);
			}
		}

		return rules;
	}
}