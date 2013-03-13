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

import java.util.List;

/**
 * An associative rule extracted from the training data
 * 
 * @author Adriano Veloso (algorithm and original C++ implementation)
 * @author Gesse Dafe (Java implementation)
 */
public class LACRule
{
	private final int predictedClass;
	private List<String> featuresLabels;
	private List<Integer> featuersIndexed;
	private String classLabel;
	
	private final Metrics metrics;

	/**
	 * Constructs a new {@link Rule}
	 * 
	 * @param support
	 * @param confidence
	 * @param predictedClass
	 */
	LACRule(Metrics metrics, int predictedClass)
	{
		this.metrics = metrics;
		this.predictedClass = predictedClass;
	}

	/**
	 * @return the support
	 */
	double getSupport()
	{
		return metrics.getMetrics()[1];
	}

	/**
	 * @return the confidence
	 */
	double getConfidence()
	{
		return metrics.getMetrics()[0];
	}

	/**
	 * @return the predictedClass
	 */
	int getPredictedClass()
	{
		return predictedClass;
	}
	
	List<Integer> getFeaturesIndexed(){
		return featuersIndexed;
	}
	
	@Override
	public String toString()
	{
		return "{features:{" + featuresLabels + "}, class:" + classLabel + ", metrics:" + this.metrics +"}\n";
	}

	void setPattern(List<String> featuresLabels)
	{
		this.featuresLabels = featuresLabels;
	}
	
	void setFeaturesIndexed(List<Integer> featuresIndexed){
		this.featuersIndexed = featuresIndexed;
	}

	void setClassLabel(String classLabel)
	{
		this.classLabel = classLabel;
	}
	
	String getClassLabel(){
		return this.classLabel;
	}
	
	int getClassIndex(){
		return this.predictedClass;
	}
	
	double[] getMetrics(){
		return metrics.getMetrics();
	}
}