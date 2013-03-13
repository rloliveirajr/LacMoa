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

/**
 * Maps features to numeric values for enhancing performance.
 * 
 * @author Adriano Veloso (algorithm and original C++ implementation)
 * @author Gesse Dafe (Java implementation)
 */
public class LACFeatureIndex implements Serializable
{
	private static final long serialVersionUID = 5460090231843051865L;

	private LACBidirectionalMap<LACFeature, Integer> indexed = new LACBidirectionalMap<LACFeature, Integer>(10000);

	/**
	 * Returns the index of the given feature
	 * 
	 * @param feature
	 */
	int indexOf(LACFeature feature)
	{
		Integer index = indexed.get(feature);
		if (index == null)
		{
			index = indexed.size();
			indexed.put(feature, index);
		}
		return index;
	}

	/**
	 * Returns a feature by its index
	 * 
	 * @param indexedFeature
	 */
	LACFeature getFeature(int indexedFeature)
	{
		return indexed.reverseGet(indexedFeature);
	}

	@Override
	public String toString()
	{
		return indexed.reverseString();
	}
}
